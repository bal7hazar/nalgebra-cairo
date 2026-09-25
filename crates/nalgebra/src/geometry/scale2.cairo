//! `Scale2`: a 2-dimensional non-uniform scale (upstream `nalgebra::Scale2`, i.e. `Scale<T, 2>`),
//! WP 8.4-P10.
//!
//! A scale is the vector of the factors multiplied to the coordinates of a point, one per axis
//! (the diagonal of its homogeneous matrix). Written from one template for the sizes 1 to 6; the
//! vector is the matching shape `Vector2`. A scale acts on POINTS (`transform_point`) and on
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
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;
use super::quaternion::ApproxEqTrait;

/// A non-uniform scale by `vector`, one factor per axis. The field name is upstream's
/// (`Scale { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Scale2<T> {
    pub vector: Vector2<T>,
}

/// Operations of `Scale2<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Scale2Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Scale2Trait<T> {
    /// The scale by `(x, y)`. Exact. Upstream: `Scale2::new`.
    #[inline(always)]
    fn new(x: T, y: T) -> Scale2<T> {
        Scale2 { vector: Vector2 { x, y } }
    }

    /// The identity scale (all factors one). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Scale2<T> {
        Scale2 { vector: Vector2 { x: R::one(), y: R::one() } }
    }

    /// The inverse scale, factor by factor (`1 / f`, one correctly rounded reciprocal each), or
    /// `None`
    /// when a factor is zero. Upstream: `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Scale2<T>) -> Option<Scale2<T>> {
        let v = self.vector;
        if v.x == R::zero() || v.y == R::zero() {
            return None;
        }
        Some(Scale2 { vector: Vector2 { x: R::recip(v.x), y: R::recip(v.y) } })
    }

    /// The inverse scale WITHOUT the zero test of `try_inverse` (upstream's `unsafe`
    /// `inverse_unchecked`; Cairo has no unsafe code, so a zero factor panics with `Fixed: division
    /// by zero` instead of being undefined behaviour). Upstream: `inverse_unchecked`.
    #[inline(always)]
    fn inverse_unchecked(self: Scale2<T>) -> Scale2<T> {
        let v = self.vector;
        Scale2 { vector: Vector2 { x: R::recip(v.x), y: R::recip(v.y) } }
    }

    /// The pseudo-inverse: every non-zero factor inverted, the zero factors kept at zero (never
    /// panics, except on the overflow of a tiny factor's reciprocal). Upstream: `pseudo_inverse`.
    #[inline(always)]
    fn pseudo_inverse(self: Scale2<T>) -> Scale2<T> {
        let v = self.vector;
        Scale2 {
            vector: Vector2 {
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
            },
        }
    }

    /// Inverts `self` in place and returns `true`, or leaves it unchanged and returns `false` when
    /// a factor is zero. Upstream: `try_inverse_mut`.
    #[inline(always)]
    fn try_inverse_mut(ref self: Scale2<T>) -> bool {
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
    fn transform_point(self: Scale2<T>, p: Point2<T>) -> Point2<T> {
        let v = self.vector;
        Point2 { x: v.x * p.x, y: v.y * p.y }
    }

    /// `self⁻¹ * p`: every coordinate of `p` divided by its factor (ONE correctly rounded
    /// division each, more accurate and cheaper than upstream's reciprocal then product), or `None`
    /// when a factor is zero. Panics on overflow. Upstream: `try_inverse_transform_point`.
    #[inline(always)]
    fn try_inverse_transform_point(self: Scale2<T>, p: Point2<T>) -> Option<Point2<T>> {
        let v = self.vector;
        if v.x == R::zero() || v.y == R::zero() {
            return None;
        }
        Some(Point2 { x: R::div(p.x, v.x), y: R::div(p.y, v.y) })
    }

    /// `self * v`: every component of the vector `v` multiplied by its factor (one floored product
    /// each). Panics on overflow. Upstream: `Mul<Vector> for Scale`.
    #[inline(always)]
    fn mul_vector(self: Scale2<T>, v: Vector2<T>) -> Vector2<T> {
        let s = self.vector;
        Vector2 { x: s.x * v.x, y: s.y * v.y }
    }

    /// `self * k`: every factor multiplied by the scalar `k` (one floored product each). Panics on
    /// overflow. Upstream: `Mul<T> for Scale`.
    #[inline(always)]
    fn scale(self: Scale2<T>, k: T) -> Scale2<T> {
        let v = self.vector;
        Scale2 { vector: Vector2 { x: v.x * k, y: v.y * k } }
    }

    /// The scale as a 3x3 homogeneous matrix: the diagonal matrix of the factors and a final 1.
    /// Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Scale2<T>) -> Matrix3<T> {
        let v = self.vector;
        Matrix3 {
            m11: v.x,
            m21: R::zero(),
            m31: R::zero(),
            m12: R::zero(),
            m22: v.y,
            m32: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
        }
    }

    /// `true` when every factor is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance being
    /// counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Scale2<T>, other: Scale2<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps) && R::abs_diff_eq(a.y, b.y, ulps)
    }

    /// `true` when every factor is `relative_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Scale2<T>, other: Scale2<T>, epsilon: u64, max_relative: T) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
    }

    /// `true` when every factor is `ulps_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Scale2<T>, other: Scale2<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
    }

    /// The same scale with every factor converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Scale<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Scale2<T>) -> Scale2<U> {
        let v = self.vector;
        Scale2 { vector: Vector2 { x: v.x.into(), y: v.y.into() } }
    }
}

/// `a * b`: the composition of two scales, the component-wise product of their factors (one
/// floored product each; scales commute). Panics on overflow. Upstream: `Mul<Scale> for Scale`.
pub impl Scale2Mul<T, +Mul<T>, +Copy<T>, +Drop<T>> of Mul<Scale2<T>> {
    #[inline(always)]
    fn mul(lhs: Scale2<T>, rhs: Scale2<T>) -> Scale2<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Scale2 { vector: Vector2 { x: a.x * b.x, y: a.y * b.y } }
    }
}

/// `self *= other`: `self * other` in place. Panics on overflow. Upstream: `MulAssign<Scale>`.
pub impl Scale2MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale2<T>, Scale2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Scale2<T>, rhs: Scale2<T>) {
        let (a, b) = (self.vector, rhs.vector);
        self = Scale2 { vector: Vector2 { x: a.x * b.x, y: a.y * b.y } };
    }
}

/// `self *= k`: every factor multiplied by the scalar `k` in place (one floored product each).
/// Panics on overflow. Upstream: `MulAssign<T>`.
pub impl Scale2MulAssignScalar<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale2<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Scale2<T>, rhs: T) {
        let a = self.vector;
        self = Scale2 { vector: Vector2 { x: a.x * rhs, y: a.y * rhs } };
    }
}

/// `One::one()`: the identity scale; `is_one` tests for all factors equal to one exactly.
/// Upstream: `num::One for Scale`.
pub impl Scale2One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Scale2<T>> {
    #[inline(always)]
    fn one() -> Scale2<T> {
        Scale2 { vector: Vector2 { x: R::one(), y: R::one() } }
    }

    #[inline(always)]
    fn is_one(self: @Scale2<T>) -> bool {
        let v = *self.vector;
        v.x == R::one() && v.y == R::one()
    }

    #[inline(always)]
    fn is_non_one(self: @Scale2<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `v.into()`: the scale by the factors `v`. Upstream: `From<Vector2> for Scale2`.
pub impl Scale2FromVector<T> of Into<Vector2<T>, Scale2<T>> {
    #[inline(always)]
    fn into(self: Vector2<T>) -> Scale2<T> {
        Scale2 { vector: self }
    }
}

/// `p.into()`: the scale whose factors are the coordinates of `p`. Upstream: `From<Point2> for
/// Scale2`.
pub impl Scale2FromPoint<T> of Into<Point2<T>, Scale2<T>> {
    #[inline(always)]
    fn into(self: Point2<T>) -> Scale2<T> {
        let Point2 { x, y } = self;
        Scale2 { vector: Vector2 { x, y } }
    }
}

/// `[x, y].into()`: the scale by these factors. Upstream: `From<[T; 2]>`.
pub impl Scale2FromArray<T> of Into<[T; 2], Scale2<T>> {
    #[inline(always)]
    fn into(self: [T; 2]) -> Scale2<T> {
        let [x, y] = self;
        Scale2 { vector: Vector2 { x, y } }
    }
}

/// The factors as an array. Upstream: `Into<[T; 2]>`.
pub impl Scale2IntoArray<T> of Into<Scale2<T>, [T; 2]> {
    #[inline(always)]
    fn into(self: Scale2<T>) -> [T; 2] {
        let Vector2 { x, y } = self.vector;
        [x, y]
    }
}

/// `s.into()`: the homogeneous matrix (`to_homogeneous`). Exact. Upstream: `From<Scale2> for
/// Matrix3`.
pub impl Matrix3FromScale2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Into<Scale2<T>, Matrix3<T>> {
    #[inline(always)]
    fn into(self: Scale2<T>) -> Matrix3<T> {
        Scale2Trait::to_homogeneous(self)
    }
}
