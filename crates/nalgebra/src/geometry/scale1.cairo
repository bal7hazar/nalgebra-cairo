//! `Scale1`: a 1-dimensional non-uniform scale (upstream `nalgebra::Scale1`, i.e. `Scale<T, 1>`),
//! WP 8.4-P10.
//!
//! A scale is the vector of the factors multiplied to the coordinates of a point, one per axis (the
//! diagonal of its homogeneous matrix). Written from one template for the sizes 1 to 6; the vector
//! is the matching shape `Matrix1`. A scale acts on POINTS (`transform_point`) and on vectors
//! (`mul_vector`) alike; its product with another scale, the product with a scalar (`scale`) and
//! the inverse are component-wise.
//!
//! Numeric contract (AGENTS.md): every product is one floored fixed-point multiplication per
//! component, every inverse one correctly rounded reciprocal (`Real::recip`, to nearest, ties to
//! even, like `f64 /`); overflow panics, nothing wraps. `inverse_unchecked` is unchecked only in
//! upstream's sense (no zero test): a zero factor panics with `Fixed: division by zero`.

use core::num::traits::One;
use core::ops::MulAssign;
use simba::scalar::Real;
use crate::base::matrix1::Matrix1;
use crate::base::matrix2::Matrix2;
use super::point1::Point1;
use super::quaternion::ApproxEqTrait;

/// A non-uniform scale by `vector`, one factor per axis. The field name is upstream's (`Scale {
/// vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Scale1<T> {
    pub vector: Matrix1<T>,
}

/// Operations of `Scale1<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Scale1Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Scale1Trait<T> {
    /// The scale by `(x)`. Exact. Upstream: `Scale1::new`.
    #[inline(always)]
    fn new(x: T) -> Scale1<T> {
        Scale1 { vector: Matrix1 { x } }
    }

    /// The identity scale (all factors one). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Scale1<T> {
        Scale1 { vector: Matrix1 { x: R::one() } }
    }

    /// The inverse scale, factor by factor (`1 / f`, one correctly rounded reciprocal each), or
    /// `None` when a factor is zero. Upstream: `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Scale1<T>) -> Option<Scale1<T>> {
        let v = self.vector;
        if v.x == R::zero() {
            return None;
        }
        Some(Scale1 { vector: Matrix1 { x: R::recip(v.x) } })
    }

    /// The inverse scale WITHOUT the zero test of `try_inverse` (upstream's `unsafe`
    /// `inverse_unchecked`; Cairo has no unsafe code, so a zero factor panics with `Fixed: division
    /// by zero` instead of being undefined behaviour). Upstream: `inverse_unchecked`.
    #[inline(always)]
    fn inverse_unchecked(self: Scale1<T>) -> Scale1<T> {
        let v = self.vector;
        Scale1 { vector: Matrix1 { x: R::recip(v.x) } }
    }

    /// The pseudo-inverse: every non-zero factor inverted, the zero factors kept at zero (never
    /// panics, except on the overflow of a tiny factor's reciprocal). Upstream: `pseudo_inverse`.
    #[inline(always)]
    fn pseudo_inverse(self: Scale1<T>) -> Scale1<T> {
        let v = self.vector;
        Scale1 { vector: Matrix1 { x: if v.x == R::zero() {
            R::zero()
        } else {
            R::recip(v.x)
        } } }
    }

    /// Inverts `self` in place and returns `true`, or leaves it unchanged and returns `false` when
    /// a factor is zero. Upstream: `try_inverse_mut`.
    #[inline(always)]
    fn try_inverse_mut(ref self: Scale1<T>) -> bool {
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
    fn transform_point(self: Scale1<T>, p: Point1<T>) -> Point1<T> {
        let v = self.vector;
        Point1 { x: v.x * p.x }
    }

    /// `self⁻¹ * p`: every coordinate of `p` divided by its factor (ONE correctly rounded
    /// division each, more accurate and cheaper than upstream's reciprocal then product), or `None`
    /// when a factor is zero. Panics on overflow. Upstream: `try_inverse_transform_point`.
    #[inline(always)]
    fn try_inverse_transform_point(self: Scale1<T>, p: Point1<T>) -> Option<Point1<T>> {
        let v = self.vector;
        if v.x == R::zero() {
            return None;
        }
        Some(Point1 { x: R::div(p.x, v.x) })
    }

    /// `self * v`: every component of the vector `v` multiplied by its factor (one floored product
    /// each). Panics on overflow. Upstream: `Mul<Vector> for Scale`.
    #[inline(always)]
    fn mul_vector(self: Scale1<T>, v: Matrix1<T>) -> Matrix1<T> {
        let s = self.vector;
        Matrix1 { x: s.x * v.x }
    }

    /// `self * k`: every factor multiplied by the scalar `k` (one floored product each). Panics on
    /// overflow. Upstream: `Mul<T> for Scale`.
    #[inline(always)]
    fn scale(self: Scale1<T>, k: T) -> Scale1<T> {
        let v = self.vector;
        Scale1 { vector: Matrix1 { x: v.x * k } }
    }

    /// The scale as a 2x2 homogeneous matrix: the diagonal matrix of the factors and a final 1.
    /// Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Scale1<T>) -> Matrix2<T> {
        let v = self.vector;
        Matrix2 { m11: v.x, m21: R::zero(), m12: R::zero(), m22: R::one() }
    }

    /// `true` when every factor is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance being
    /// counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Scale1<T>, other: Scale1<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
    }

    /// `true` when every factor is `relative_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Scale1<T>, other: Scale1<T>, epsilon: u64, max_relative: T) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
    }

    /// `true` when every factor is `ulps_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Scale1<T>, other: Scale1<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
    }

    /// The same scale with every factor converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Scale<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Scale1<T>) -> Scale1<U> {
        let v = self.vector;
        Scale1 { vector: Matrix1 { x: v.x.into() } }
    }
}

/// `a * b`: the composition of two scales, the component-wise product of their factors (one floored
/// product each; scales commute). Panics on overflow. Upstream: `Mul<Scale> for Scale`.
pub impl Scale1Mul<T, +Mul<T>, +Copy<T>, +Drop<T>> of Mul<Scale1<T>> {
    #[inline(always)]
    fn mul(lhs: Scale1<T>, rhs: Scale1<T>) -> Scale1<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Scale1 { vector: Matrix1 { x: a.x * b.x } }
    }
}

/// `self *= other`: `self * other` in place. Panics on overflow. Upstream: `MulAssign<Scale>`.
pub impl Scale1MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale1<T>, Scale1<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Scale1<T>, rhs: Scale1<T>) {
        let (a, b) = (self.vector, rhs.vector);
        self = Scale1 { vector: Matrix1 { x: a.x * b.x } };
    }
}

/// `self *= k`: every factor multiplied by the scalar `k` in place (one floored product each).
/// Panics on overflow. Upstream: `MulAssign<T>`.
pub impl Scale1MulAssignScalar<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale1<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Scale1<T>, rhs: T) {
        let a = self.vector;
        self = Scale1 { vector: Matrix1 { x: a.x * rhs } };
    }
}

/// `One::one()`: the identity scale; `is_one` tests for all factors equal to one exactly. Upstream:
/// `num::One for Scale`.
pub impl Scale1One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Scale1<T>> {
    #[inline(always)]
    fn one() -> Scale1<T> {
        Scale1 { vector: Matrix1 { x: R::one() } }
    }

    #[inline(always)]
    fn is_one(self: @Scale1<T>) -> bool {
        let v = *self.vector;
        v.x == R::one()
    }

    #[inline(always)]
    fn is_non_one(self: @Scale1<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `v.into()`: the scale by the factors `v`. Upstream: `From<Matrix1> for Scale1`.
pub impl Scale1FromVector<T> of Into<Matrix1<T>, Scale1<T>> {
    #[inline(always)]
    fn into(self: Matrix1<T>) -> Scale1<T> {
        Scale1 { vector: self }
    }
}

/// `p.into()`: the scale whose factors are the coordinates of `p`. Upstream: `From<Point1> for
/// Scale1`.
pub impl Scale1FromPoint<T> of Into<Point1<T>, Scale1<T>> {
    #[inline(always)]
    fn into(self: Point1<T>) -> Scale1<T> {
        let Point1 { x } = self;
        Scale1 { vector: Matrix1 { x } }
    }
}

/// `[x].into()`: the scale by these factors. Upstream: `From<[T; 1]>`.
pub impl Scale1FromArray<T> of Into<[T; 1], Scale1<T>> {
    #[inline(always)]
    fn into(self: [T; 1]) -> Scale1<T> {
        let [x] = self;
        Scale1 { vector: Matrix1 { x } }
    }
}

/// The factors as an array. Upstream: `Into<[T; 1]>`.
pub impl Scale1IntoArray<T> of Into<Scale1<T>, [T; 1]> {
    #[inline(always)]
    fn into(self: Scale1<T>) -> [T; 1] {
        let Matrix1 { x } = self.vector;
        [x]
    }
}

/// `s.into()`: the homogeneous matrix (`to_homogeneous`). Exact. Upstream: `From<Scale1> for
/// Matrix2`.
pub impl Matrix2FromScale1<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Into<Scale1<T>, Matrix2<T>> {
    #[inline(always)]
    fn into(self: Scale1<T>) -> Matrix2<T> {
        Scale1Trait::to_homogeneous(self)
    }
}
