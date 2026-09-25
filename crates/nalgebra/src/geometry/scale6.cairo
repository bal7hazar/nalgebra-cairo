//! `Scale6`: a 6-dimensional non-uniform scale (upstream `nalgebra::Scale6`, i.e. `Scale<T, 6>`),
//! WP 8.4-P10.
//!
//! A scale is the vector of the factors multiplied to the coordinates of a point, one per axis
//! (the diagonal of its homogeneous matrix). Written from one template for the sizes 1 to 6; the
//! vector is the matching shape `Vector6`. A scale acts on POINTS (`transform_point`) and on
//! vectors (`mul_vector`) alike; its product with another scale, the product with a scalar
//! (`scale`) and the inverse are component-wise.
//!
//! Numeric contract (AGENTS.md): every product is one floored fixed-point multiplication per
//! component, every inverse one correctly rounded reciprocal (`Real::recip`, to nearest, ties to
//! even, like `f64 /`); overflow panics, nothing wraps. `inverse_unchecked` is unchecked only
//! in upstream's sense (no zero test): a zero factor panics with `Fixed: division by zero`.
//!
//! There is no `to_homogeneous` (nor `From<Scale6> for Matrix7`): the homogeneous matrix of a 6D
//! scale is 7x7, and the static shapes stop at 6 (DESIGN D4), like `Translation6`.

use core::num::traits::One;
use core::ops::MulAssign;
use simba::scalar::Real;
use crate::base::vector6::Vector6;
use super::point6::Point6;
use super::quaternion::ApproxEqTrait;

/// A non-uniform scale by `vector`, one factor per axis. The field name is upstream's
/// (`Scale { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Scale6<T> {
    pub vector: Vector6<T>,
}

/// Operations of `Scale6<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Scale6Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Scale6Trait<T> {
    /// The scale by `(x, y, z, w, a, b)`. Exact. Upstream: `Scale6::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T, a: T, b: T) -> Scale6<T> {
        Scale6 { vector: Vector6 { x, y, z, w, a, b } }
    }

    /// The identity scale (all factors one). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Scale6<T> {
        Scale6 {
            vector: Vector6 {
                x: R::one(), y: R::one(), z: R::one(), w: R::one(), a: R::one(), b: R::one(),
            },
        }
    }

    /// The inverse scale, factor by factor (`1 / f`, one correctly rounded reciprocal each), or
    /// `None`
    /// when a factor is zero. Upstream: `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Scale6<T>) -> Option<Scale6<T>> {
        let v = self.vector;
        if v.x == R::zero()
            || v.y == R::zero()
            || v.z == R::zero()
            || v.w == R::zero()
            || v.a == R::zero()
            || v.b == R::zero() {
            return None;
        }
        Some(
            Scale6 {
                vector: Vector6 {
                    x: R::recip(v.x),
                    y: R::recip(v.y),
                    z: R::recip(v.z),
                    w: R::recip(v.w),
                    a: R::recip(v.a),
                    b: R::recip(v.b),
                },
            },
        )
    }

    /// The inverse scale WITHOUT the zero test of `try_inverse` (upstream's `unsafe`
    /// `inverse_unchecked`; Cairo has no unsafe code, so a zero factor panics with `Fixed: division
    /// by zero` instead of being undefined behaviour). Upstream: `inverse_unchecked`.
    #[inline(always)]
    fn inverse_unchecked(self: Scale6<T>) -> Scale6<T> {
        let v = self.vector;
        Scale6 {
            vector: Vector6 {
                x: R::recip(v.x),
                y: R::recip(v.y),
                z: R::recip(v.z),
                w: R::recip(v.w),
                a: R::recip(v.a),
                b: R::recip(v.b),
            },
        }
    }

    /// The pseudo-inverse: every non-zero factor inverted, the zero factors kept at zero (never
    /// panics, except on the overflow of a tiny factor's reciprocal). Upstream: `pseudo_inverse`.
    #[inline(always)]
    fn pseudo_inverse(self: Scale6<T>) -> Scale6<T> {
        let v = self.vector;
        Scale6 {
            vector: Vector6 {
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
                a: if v.a == R::zero() {
                    R::zero()
                } else {
                    R::recip(v.a)
                },
                b: if v.b == R::zero() {
                    R::zero()
                } else {
                    R::recip(v.b)
                },
            },
        }
    }

    /// Inverts `self` in place and returns `true`, or leaves it unchanged and returns `false` when
    /// a factor is zero. Upstream: `try_inverse_mut`.
    #[inline(always)]
    fn try_inverse_mut(ref self: Scale6<T>) -> bool {
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
    fn transform_point(self: Scale6<T>, p: Point6<T>) -> Point6<T> {
        let v = self.vector;
        Point6 {
            x: v.x * p.x, y: v.y * p.y, z: v.z * p.z, w: v.w * p.w, a: v.a * p.a, b: v.b * p.b,
        }
    }

    /// `self⁻¹ * p`: every coordinate of `p` divided by its factor (ONE correctly rounded
    /// division each, more accurate and cheaper than upstream's reciprocal then product), or `None`
    /// when a factor is zero. Panics on overflow. Upstream: `try_inverse_transform_point`.
    #[inline(always)]
    fn try_inverse_transform_point(self: Scale6<T>, p: Point6<T>) -> Option<Point6<T>> {
        let v = self.vector;
        if v.x == R::zero()
            || v.y == R::zero()
            || v.z == R::zero()
            || v.w == R::zero()
            || v.a == R::zero()
            || v.b == R::zero() {
            return None;
        }
        Some(
            Point6 {
                x: R::div(p.x, v.x),
                y: R::div(p.y, v.y),
                z: R::div(p.z, v.z),
                w: R::div(p.w, v.w),
                a: R::div(p.a, v.a),
                b: R::div(p.b, v.b),
            },
        )
    }

    /// `self * v`: every component of the vector `v` multiplied by its factor (one floored product
    /// each). Panics on overflow. Upstream: `Mul<Vector> for Scale`.
    #[inline(always)]
    fn mul_vector(self: Scale6<T>, v: Vector6<T>) -> Vector6<T> {
        let s = self.vector;
        Vector6 {
            x: s.x * v.x, y: s.y * v.y, z: s.z * v.z, w: s.w * v.w, a: s.a * v.a, b: s.b * v.b,
        }
    }

    /// `self * k`: every factor multiplied by the scalar `k` (one floored product each). Panics on
    /// overflow. Upstream: `Mul<T> for Scale`.
    #[inline(always)]
    fn scale(self: Scale6<T>, k: T) -> Scale6<T> {
        let v = self.vector;
        Scale6 {
            vector: Vector6 {
                x: v.x * k, y: v.y * k, z: v.z * k, w: v.w * k, a: v.a * k, b: v.b * k,
            },
        }
    }

    /// `true` when every factor is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance being
    /// counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Scale6<T>, other: Scale6<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
            && R::abs_diff_eq(a.y, b.y, ulps)
            && R::abs_diff_eq(a.z, b.z, ulps)
            && R::abs_diff_eq(a.w, b.w, ulps)
            && R::abs_diff_eq(a.a, b.a, ulps)
            && R::abs_diff_eq(a.b, b.b, ulps)
    }

    /// `true` when every factor is `relative_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Scale6<T>, other: Scale6<T>, epsilon: u64, max_relative: T) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.z, b.z, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.w, b.w, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.a, b.a, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.b, b.b, epsilon, max_relative)
    }

    /// `true` when every factor is `ulps_eq` to the matching factor of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Scale6<T>, other: Scale6<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.z, b.z, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.w, b.w, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.a, b.a, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.b, b.b, epsilon, max_ulps)
    }

    /// The same scale with every factor converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Scale<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Scale6<T>) -> Scale6<U> {
        let v = self.vector;
        Scale6 {
            vector: Vector6 {
                x: v.x.into(),
                y: v.y.into(),
                z: v.z.into(),
                w: v.w.into(),
                a: v.a.into(),
                b: v.b.into(),
            },
        }
    }
}

/// `a * b`: the composition of two scales, the component-wise product of their factors (one
/// floored product each; scales commute). Panics on overflow. Upstream: `Mul<Scale> for Scale`.
pub impl Scale6Mul<T, +Mul<T>, +Copy<T>, +Drop<T>> of Mul<Scale6<T>> {
    #[inline(always)]
    fn mul(lhs: Scale6<T>, rhs: Scale6<T>) -> Scale6<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Scale6 {
            vector: Vector6 {
                x: a.x * b.x, y: a.y * b.y, z: a.z * b.z, w: a.w * b.w, a: a.a * b.a, b: a.b * b.b,
            },
        }
    }
}

/// `self *= other`: `self * other` in place. Panics on overflow. Upstream: `MulAssign<Scale>`.
pub impl Scale6MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale6<T>, Scale6<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Scale6<T>, rhs: Scale6<T>) {
        let (a, b) = (self.vector, rhs.vector);
        self =
            Scale6 {
                vector: Vector6 {
                    x: a.x * b.x,
                    y: a.y * b.y,
                    z: a.z * b.z,
                    w: a.w * b.w,
                    a: a.a * b.a,
                    b: a.b * b.b,
                },
            };
    }
}

/// `self *= k`: every factor multiplied by the scalar `k` in place (one floored product each).
/// Panics on overflow. Upstream: `MulAssign<T>`.
pub impl Scale6MulAssignScalar<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Scale6<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Scale6<T>, rhs: T) {
        let a = self.vector;
        self =
            Scale6 {
                vector: Vector6 {
                    x: a.x * rhs,
                    y: a.y * rhs,
                    z: a.z * rhs,
                    w: a.w * rhs,
                    a: a.a * rhs,
                    b: a.b * rhs,
                },
            };
    }
}

/// `One::one()`: the identity scale; `is_one` tests for all factors equal to one exactly.
/// Upstream: `num::One for Scale`.
pub impl Scale6One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Scale6<T>> {
    #[inline(always)]
    fn one() -> Scale6<T> {
        Scale6 {
            vector: Vector6 {
                x: R::one(), y: R::one(), z: R::one(), w: R::one(), a: R::one(), b: R::one(),
            },
        }
    }

    #[inline(always)]
    fn is_one(self: @Scale6<T>) -> bool {
        let v = *self.vector;
        v.x == R::one()
            && v.y == R::one()
            && v.z == R::one()
            && v.w == R::one()
            && v.a == R::one()
            && v.b == R::one()
    }

    #[inline(always)]
    fn is_non_one(self: @Scale6<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `v.into()`: the scale by the factors `v`. Upstream: `From<Vector6> for Scale6`.
pub impl Scale6FromVector<T> of Into<Vector6<T>, Scale6<T>> {
    #[inline(always)]
    fn into(self: Vector6<T>) -> Scale6<T> {
        Scale6 { vector: self }
    }
}

/// `p.into()`: the scale whose factors are the coordinates of `p`. Upstream: `From<Point6> for
/// Scale6`.
pub impl Scale6FromPoint<T> of Into<Point6<T>, Scale6<T>> {
    #[inline(always)]
    fn into(self: Point6<T>) -> Scale6<T> {
        let Point6 { x, y, z, w, a, b } = self;
        Scale6 { vector: Vector6 { x, y, z, w, a, b } }
    }
}

/// `[x, y, z, w, a, b].into()`: the scale by these factors. Upstream: `From<[T; 6]>`.
pub impl Scale6FromArray<T> of Into<[T; 6], Scale6<T>> {
    #[inline(always)]
    fn into(self: [T; 6]) -> Scale6<T> {
        let [x, y, z, w, a, b] = self;
        Scale6 { vector: Vector6 { x, y, z, w, a, b } }
    }
}

/// The factors as an array. Upstream: `Into<[T; 6]>`.
pub impl Scale6IntoArray<T> of Into<Scale6<T>, [T; 6]> {
    #[inline(always)]
    fn into(self: Scale6<T>) -> [T; 6] {
        let Vector6 { x, y, z, w, a, b } = self.vector;
        [x, y, z, w, a, b]
    }
}
