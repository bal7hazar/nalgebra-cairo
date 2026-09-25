//! `Scale4`: a 4-dimensional non-uniform scale (upstream `nalgebra::Scale4`, i.e. `Scale<T, 4>`),
//! WP 8.4-P10.
//!
//! A scale is the vector of the factors multiplied to the coordinates of a point, one per axis
//! (the diagonal of its homogeneous matrix). Written from one template for the sizes 1 to 6; the
//! vector is the matching shape `Vector4`. A scale acts on POINTS (`transform_point`) and on
//! vectors (`mul_vector`) alike; its product with another scale, the product with a scalar
//! (`scale`) and the inverse are component-wise.
//!
//! Numeric contract (AGENTS.md): every product is one floored fixed-point multiplication per
//! component, every inverse one correctly rounded reciprocal (`Real::recip`, to nearest, ties to
//! even, like `f64 /`); overflow panics, nothing wraps. `inverse_unchecked` is unchecked only
//! in upstream's sense (no zero test): a zero factor panics with `Fixed: division by zero`.

use core::num::traits::One;
use core::ops::MulAssign;
use simba::scalar::Real;
use crate::base::matrix5::Matrix5;
use crate::base::vector4::Vector4;
use super::point4::Point4;
use super::quaternion::ApproxEqTrait;

/// A non-uniform scale by `vector`, one factor per axis. The field name is upstream's
/// (`Scale { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Scale4<T> {
    pub vector: Vector4<T>,
}

/// Operations of `Scale4<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Scale4Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Scale4Trait<T> {
    /// The scale by `(x, y, z, w)`. Exact. Upstream: `Scale4::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T) -> Scale4<T> {
        Scale4 { vector: Vector4 { x, y, z, w } }
    }

    /// The identity scale (all factors one). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Scale4<T> {
        Scale4 { vector: Vector4 { x: R::one(), y: R::one(), z: R::one(), w: R::one() } }
    }

    /// The inverse scale, factor by factor (`1 / f`, one correctly rounded reciprocal each), or
    /// `None`
    /// when a factor is zero. Upstream: `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Scale4<T>) -> Option<Scale4<T>> {
        let v = self.vector;
        if v.x == R::zero() || v.y == R::zero() || v.z == R::zero() || v.w == R::zero() {
            return None;
        }
        Some(
            Scale4 {
                vector: Vector4 {
                    x: R::recip(v.x), y: R::recip(v.y), z: R::recip(v.z), w: R::recip(v.w),
                },
            },
        )
    }

    /// The inverse scale WITHOUT the zero test of `try_inverse` (upstream's `unsafe`
    /// `inverse_unchecked`; Cairo has no unsafe code, so a zero factor panics with `Fixed: division
    /// by zero` instead of being undefined behaviour). Upstream: `inverse_unchecked`.
    #[inline(always)]
    fn inverse_unchecked(self: Scale4<T>) -> Scale4<T> {
        let v = self.vector;
        Scale4 {
            vector: Vector4 {
                x: R::recip(v.x), y: R::recip(v.y), z: R::recip(v.z), w: R::recip(v.w),
            },
        }
    }

    /// The pseudo-inverse: every non-zero factor inverted, the zero factors kept at zero (never
    /// panics, except on the overflow of a tiny factor's reciprocal). Upstream: `pseudo_inverse`.
    #[inline(always)]
    fn pseudo_inverse(self: Scale4<T>) -> Scale4<T> {
        let v = self.vector;
        Scale4 {
            vector: Vector4 {
                x: if v.x == R::zero() {
                    R::zero()
                } else {
                    R::recip(v.x)
                },
                y: if v.y == R::zero() {
                    R::zero()
                } else {
                    R::recip(v.y)
                },
                z: if v.z == R::zero() {
                    R::zero()
                } else {
                    R::recip(v.z)
                },
                w: if v.w == R::zero() {
                    R::zero()
                } else {
                    R::recip(v.w)
                },
            },
        }
    }

    /// Inverts `self` in place and returns `true`, or leaves it unchanged and returns `false` when
    /// a factor is zero. Upstream: `try_inverse_mut`.
    #[inline(always)]
    fn try_inverse_mut(ref self: Scale4<T>) -> bool {
        match Self::try_inverse(self) {
            Some(inverse) => {
                self = inverse;
                true
            },
            None => false,
        }
    }

    /// `self * p`: every coordinate of `p` multiplied by its factor (one floored product each).
    /// Panics on overflow. Upstream: `transform_point` (`s * p`).
    #[inline(always)]
    fn transform_point(self: Scale4<T>, p: Point4<T>) -> Point4<T> {
        let v = self.vector;
        Point4 { x: v.x * p.x, y: v.y * p.y, z: v.z * p.z, w: v.w * p.w }
    }

    /// `self⁻¹ * p`: every coordinate of `p` divided by its factor (ONE correctly rounded
    /// division each, more accurate and cheaper than upstream's reciprocal then product), or `None`
    /// when a factor is zero. Panics on overflow. Upstream: `try_inverse_transform_point`.
    #[inline(always)]
    fn try_inverse_transform_point(self: Scale4<T>, p: Point4<T>) -> Option<Point4<T>> {
        let v = self.vector;
        if v.x == R::zero() || v.y == R::zero() || v.z == R::zero() || v.w == R::zero() {
            return None;
        }
        Some(
            Point4 {
                x: R::div(p.x, v.x), y: R::div(p.y, v.y), z: R::div(p.z, v.z), w: R::div(p.w, v.w),
            },
        )
    }

    /// `self * v`: every component of the vector `v` multiplied by its factor (one floored product
    /// each). Panics on overflow. Upstream: `Mul<Vector> for Scale`.
    #[inline(always)]
    fn mul_vector(self: Scale4<T>, v: Vector4<T>) -> Vector4<T> {
        let s = self.vector;
        Vector4 { x: s.x * v.x, y: s.y * v.y, z: s.z * v.z, w: s.w * v.w }
    }

    /// `self * k`: every factor multiplied by the scalar `k` (one floored product each). Panics on
    /// overflow. Upstream: `Mul<T> for Scale`.
    #[inline(always)]
    fn scale(self: Scale4<T>, k: T) -> Scale4<T> {
        let v = self.vector;
        Scale4 { vector: Vector4 { x: v.x * k, y: v.y * k, z: v.z * k, w: v.w * k } }
    }

    /// The scale as a 5x5 homogeneous matrix: the diagonal matrix of the factors and a final 1.
    /// Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Scale4<T>) -> Matrix5<T> {
        let v = self.vector;
        Matrix5 {
            m11: v.x,
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m51: R::zero(),
            m12: R::zero(),
            m22: v.y,
            m32: R::zero(),
            m42: R::zero(),
            m52: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: v.z,
            m43: R::zero(),
            m53: R::zero(),
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: v.w,
            m54: R::zero(),
            m15: R::zero(),
            m25: R::zero(),
            m35: R::zero(),
            m45: R::zero(),
            m55: R::one(),
        }
    }

    /// `true` when every factor is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance being
    /// counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Scale4<T>, other: Scale4<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
            && R::abs_diff_eq(a.y, b.y, ulps)
            && R::abs_diff_eq(a.z, b.z, ulps)
            && R::abs_diff_eq(a.w, b.w, ulps)
    }

    /// `true` when every factor is `relative_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Scale4<T>, other: Scale4<T>, epsilon: u64, max_relative: T) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.z, b.z, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.w, b.w, epsilon, max_relative)
    }

    /// `true` when every factor is `ulps_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Scale4<T>, other: Scale4<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.z, b.z, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.w, b.w, epsilon, max_ulps)
    }

    /// The same scale with every factor converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Scale<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Scale4<T>) -> Scale4<U> {
        let v = self.vector;
        Scale4 { vector: Vector4 { x: v.x.into(), y: v.y.into(), z: v.z.into(), w: v.w.into() } }
    }
}

/// `a * b`: the composition of two scales, the component-wise product of their factors (one
/// floored product each; scales commute). Panics on overflow. Upstream: `Mul<Scale> for Scale`.
pub impl Scale4Mul<T, +Mul<T>, +Copy<T>, +Drop<T>> of Mul<Scale4<T>> {
    #[inline(always)]
    fn mul(lhs: Scale4<T>, rhs: Scale4<T>) -> Scale4<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Scale4 { vector: Vector4 { x: a.x * b.x, y: a.y * b.y, z: a.z * b.z, w: a.w * b.w } }
    }
}

/// `self *= other`: `self * other` in place. Panics on overflow. Upstream: `MulAssign<Scale>`.
pub impl Scale4MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale4<T>, Scale4<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Scale4<T>, rhs: Scale4<T>) {
        let (a, b) = (self.vector, rhs.vector);
        self =
            Scale4 { vector: Vector4 { x: a.x * b.x, y: a.y * b.y, z: a.z * b.z, w: a.w * b.w } };
    }
}

/// `self *= k`: every factor multiplied by the scalar `k` in place (one floored product each).
/// Panics on overflow. Upstream: `MulAssign<T>`.
pub impl Scale4MulAssignScalar<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale4<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Scale4<T>, rhs: T) {
        let a = self.vector;
        self =
            Scale4 { vector: Vector4 { x: a.x * rhs, y: a.y * rhs, z: a.z * rhs, w: a.w * rhs } };
    }
}

/// `One::one()`: the identity scale; `is_one` tests for all factors equal to one exactly.
/// Upstream: `num::One for Scale`.
pub impl Scale4One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Scale4<T>> {
    #[inline(always)]
    fn one() -> Scale4<T> {
        Scale4 { vector: Vector4 { x: R::one(), y: R::one(), z: R::one(), w: R::one() } }
    }

    #[inline(always)]
    fn is_one(self: @Scale4<T>) -> bool {
        let v = *self.vector;
        v.x == R::one() && v.y == R::one() && v.z == R::one() && v.w == R::one()
    }

    #[inline(always)]
    fn is_non_one(self: @Scale4<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `v.into()`: the scale by the factors `v`. Upstream: `From<Vector4> for Scale4`.
pub impl Scale4FromVector<T> of Into<Vector4<T>, Scale4<T>> {
    #[inline(always)]
    fn into(self: Vector4<T>) -> Scale4<T> {
        Scale4 { vector: self }
    }
}

/// `p.into()`: the scale whose factors are the coordinates of `p`. Upstream: `From<Point4> for
/// Scale4`.
pub impl Scale4FromPoint<T> of Into<Point4<T>, Scale4<T>> {
    #[inline(always)]
    fn into(self: Point4<T>) -> Scale4<T> {
        let Point4 { x, y, z, w } = self;
        Scale4 { vector: Vector4 { x, y, z, w } }
    }
}

/// `[x, y, z, w].into()`: the scale by these factors. Upstream: `From<[T; 4]>`.
pub impl Scale4FromArray<T> of Into<[T; 4], Scale4<T>> {
    #[inline(always)]
    fn into(self: [T; 4]) -> Scale4<T> {
        let [x, y, z, w] = self;
        Scale4 { vector: Vector4 { x, y, z, w } }
    }
}

/// The factors as an array. Upstream: `Into<[T; 4]>`.
pub impl Scale4IntoArray<T> of Into<Scale4<T>, [T; 4]> {
    #[inline(always)]
    fn into(self: Scale4<T>) -> [T; 4] {
        let Vector4 { x, y, z, w } = self.vector;
        [x, y, z, w]
    }
}

/// `s.into()`: the homogeneous matrix (`to_homogeneous`). Exact. Upstream: `From<Scale4> for
/// Matrix5`.
pub impl Matrix5FromScale4<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Into<Scale4<T>, Matrix5<T>> {
    #[inline(always)]
    fn into(self: Scale4<T>) -> Matrix5<T> {
        Scale4Trait::to_homogeneous(self)
    }
}
