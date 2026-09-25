//! `Scale3`: a 3-dimensional non-uniform scale (upstream `nalgebra::Scale3`, i.e. `Scale<T, 3>`),
//! WP 8.4-P10.
//!
//! A scale is the vector of the factors multiplied to the coordinates of a point, one per axis
//! (the diagonal of its homogeneous matrix). Written from one template for the sizes 1 to 6; the
//! vector is the matching shape `Vector3`. A scale acts on POINTS (`transform_point`) and on
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
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;
use super::quaternion::ApproxEqTrait;

/// A non-uniform scale by `vector`, one factor per axis. The field name is upstream's
/// (`Scale { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Scale3<T> {
    pub vector: Vector3<T>,
}

/// Operations of `Scale3<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Scale3Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Scale3Trait<T> {
    /// The scale by `(x, y, z)`. Exact. Upstream: `Scale3::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T) -> Scale3<T> {
        Scale3 { vector: Vector3 { x, y, z } }
    }

    /// The identity scale (all factors one). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Scale3<T> {
        Scale3 { vector: Vector3 { x: R::one(), y: R::one(), z: R::one() } }
    }

    /// The inverse scale, factor by factor (`1 / f`, one correctly rounded reciprocal each), or
    /// `None`
    /// when a factor is zero. Upstream: `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Scale3<T>) -> Option<Scale3<T>> {
        let v = self.vector;
        if v.x == R::zero() || v.y == R::zero() || v.z == R::zero() {
            return None;
        }
        Some(Scale3 { vector: Vector3 { x: R::recip(v.x), y: R::recip(v.y), z: R::recip(v.z) } })
    }

    /// The inverse scale WITHOUT the zero test of `try_inverse` (upstream's `unsafe`
    /// `inverse_unchecked`; Cairo has no unsafe code, so a zero factor panics with `Fixed: division
    /// by zero` instead of being undefined behaviour). Upstream: `inverse_unchecked`.
    #[inline(always)]
    fn inverse_unchecked(self: Scale3<T>) -> Scale3<T> {
        let v = self.vector;
        Scale3 { vector: Vector3 { x: R::recip(v.x), y: R::recip(v.y), z: R::recip(v.z) } }
    }

    /// The pseudo-inverse: every non-zero factor inverted, the zero factors kept at zero (never
    /// panics, except on the overflow of a tiny factor's reciprocal). Upstream: `pseudo_inverse`.
    #[inline(always)]
    fn pseudo_inverse(self: Scale3<T>) -> Scale3<T> {
        let v = self.vector;
        Scale3 {
            vector: Vector3 {
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
            },
        }
    }

    /// Inverts `self` in place and returns `true`, or leaves it unchanged and returns `false` when
    /// a factor is zero. Upstream: `try_inverse_mut`.
    #[inline(always)]
    fn try_inverse_mut(ref self: Scale3<T>) -> bool {
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
    fn transform_point(self: Scale3<T>, p: Point3<T>) -> Point3<T> {
        let v = self.vector;
        Point3 { x: v.x * p.x, y: v.y * p.y, z: v.z * p.z }
    }

    /// `self⁻¹ * p`: every coordinate of `p` divided by its factor (ONE correctly rounded
    /// division each, more accurate and cheaper than upstream's reciprocal then product), or `None`
    /// when a factor is zero. Panics on overflow. Upstream: `try_inverse_transform_point`.
    #[inline(always)]
    fn try_inverse_transform_point(self: Scale3<T>, p: Point3<T>) -> Option<Point3<T>> {
        let v = self.vector;
        if v.x == R::zero() || v.y == R::zero() || v.z == R::zero() {
            return None;
        }
        Some(Point3 { x: R::div(p.x, v.x), y: R::div(p.y, v.y), z: R::div(p.z, v.z) })
    }

    /// `self * v`: every component of the vector `v` multiplied by its factor (one floored product
    /// each). Panics on overflow. Upstream: `Mul<Vector> for Scale`.
    #[inline(always)]
    fn mul_vector(self: Scale3<T>, v: Vector3<T>) -> Vector3<T> {
        let s = self.vector;
        Vector3 { x: s.x * v.x, y: s.y * v.y, z: s.z * v.z }
    }

    /// `self * k`: every factor multiplied by the scalar `k` (one floored product each). Panics on
    /// overflow. Upstream: `Mul<T> for Scale`.
    #[inline(always)]
    fn scale(self: Scale3<T>, k: T) -> Scale3<T> {
        let v = self.vector;
        Scale3 { vector: Vector3 { x: v.x * k, y: v.y * k, z: v.z * k } }
    }

    /// The scale as a 4x4 homogeneous matrix: the diagonal matrix of the factors and a final 1.
    /// Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Scale3<T>) -> Matrix4<T> {
        let v = self.vector;
        Matrix4 {
            m11: v.x,
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m12: R::zero(),
            m22: v.y,
            m32: R::zero(),
            m42: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: v.z,
            m43: R::zero(),
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: R::one(),
        }
    }

    /// `true` when every factor is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance being
    /// counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Scale3<T>, other: Scale3<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
            && R::abs_diff_eq(a.y, b.y, ulps)
            && R::abs_diff_eq(a.z, b.z, ulps)
    }

    /// `true` when every factor is `relative_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Scale3<T>, other: Scale3<T>, epsilon: u64, max_relative: T) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.z, b.z, epsilon, max_relative)
    }

    /// `true` when every factor is `ulps_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Scale3<T>, other: Scale3<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.z, b.z, epsilon, max_ulps)
    }

    /// The same scale with every factor converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Scale<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Scale3<T>) -> Scale3<U> {
        let v = self.vector;
        Scale3 { vector: Vector3 { x: v.x.into(), y: v.y.into(), z: v.z.into() } }
    }
}

/// `a * b`: the composition of two scales, the component-wise product of their factors (one
/// floored product each; scales commute). Panics on overflow. Upstream: `Mul<Scale> for Scale`.
pub impl Scale3Mul<T, +Mul<T>, +Copy<T>, +Drop<T>> of Mul<Scale3<T>> {
    #[inline(always)]
    fn mul(lhs: Scale3<T>, rhs: Scale3<T>) -> Scale3<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Scale3 { vector: Vector3 { x: a.x * b.x, y: a.y * b.y, z: a.z * b.z } }
    }
}

/// `self *= other`: `self * other` in place. Panics on overflow. Upstream: `MulAssign<Scale>`.
pub impl Scale3MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale3<T>, Scale3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Scale3<T>, rhs: Scale3<T>) {
        let (a, b) = (self.vector, rhs.vector);
        self = Scale3 { vector: Vector3 { x: a.x * b.x, y: a.y * b.y, z: a.z * b.z } };
    }
}

/// `self *= k`: every factor multiplied by the scalar `k` in place (one floored product each).
/// Panics on overflow. Upstream: `MulAssign<T>`.
pub impl Scale3MulAssignScalar<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale3<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Scale3<T>, rhs: T) {
        let a = self.vector;
        self = Scale3 { vector: Vector3 { x: a.x * rhs, y: a.y * rhs, z: a.z * rhs } };
    }
}

/// `One::one()`: the identity scale; `is_one` tests for all factors equal to one exactly.
/// Upstream: `num::One for Scale`.
pub impl Scale3One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Scale3<T>> {
    #[inline(always)]
    fn one() -> Scale3<T> {
        Scale3 { vector: Vector3 { x: R::one(), y: R::one(), z: R::one() } }
    }

    #[inline(always)]
    fn is_one(self: @Scale3<T>) -> bool {
        let v = *self.vector;
        v.x == R::one() && v.y == R::one() && v.z == R::one()
    }

    #[inline(always)]
    fn is_non_one(self: @Scale3<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `v.into()`: the scale by the factors `v`. Upstream: `From<Vector3> for Scale3`.
pub impl Scale3FromVector<T> of Into<Vector3<T>, Scale3<T>> {
    #[inline(always)]
    fn into(self: Vector3<T>) -> Scale3<T> {
        Scale3 { vector: self }
    }
}

/// `p.into()`: the scale whose factors are the coordinates of `p`. Upstream: `From<Point3> for
/// Scale3`.
pub impl Scale3FromPoint<T> of Into<Point3<T>, Scale3<T>> {
    #[inline(always)]
    fn into(self: Point3<T>) -> Scale3<T> {
        let Point3 { x, y, z } = self;
        Scale3 { vector: Vector3 { x, y, z } }
    }
}

/// `[x, y, z].into()`: the scale by these factors. Upstream: `From<[T; 3]>`.
pub impl Scale3FromArray<T> of Into<[T; 3], Scale3<T>> {
    #[inline(always)]
    fn into(self: [T; 3]) -> Scale3<T> {
        let [x, y, z] = self;
        Scale3 { vector: Vector3 { x, y, z } }
    }
}

/// The factors as an array. Upstream: `Into<[T; 3]>`.
pub impl Scale3IntoArray<T> of Into<Scale3<T>, [T; 3]> {
    #[inline(always)]
    fn into(self: Scale3<T>) -> [T; 3] {
        let Vector3 { x, y, z } = self.vector;
        [x, y, z]
    }
}

/// `s.into()`: the homogeneous matrix (`to_homogeneous`). Exact. Upstream: `From<Scale3> for
/// Matrix4`.
pub impl Matrix4FromScale3<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Into<Scale3<T>, Matrix4<T>> {
    #[inline(always)]
    fn into(self: Scale3<T>) -> Matrix4<T> {
        Scale3Trait::to_homogeneous(self)
    }
}
