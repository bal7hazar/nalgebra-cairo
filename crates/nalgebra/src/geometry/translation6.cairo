//! `Translation6`: a 6-dimensional translation (upstream `nalgebra::Translation6`, i.e.
//! `Translation<T, 6>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector6`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.
//!
//! There is no `to_homogeneous`: the homogeneous matrix of a 6D translation is 7x7, and the static
//! shapes stop at 6 (DESIGN D4).

use core::num::traits::One;
use simba::scalar::Real;
use crate::base::vector6::Vector6;
use super::point6::Point6;
use super::quaternion::ApproxEqTrait;

/// A translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation6<T> {
    pub vector: Vector6<T>,
}

/// Operations of `Translation6<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation6Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Translation6Trait<T> {
    /// The translation by `(x, y, z, w, a, b)`. Exact. Upstream: `Translation6::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T, a: T, b: T) -> Translation6<T> {
        Translation6 { vector: Vector6 { x, y, z, w, a, b } }
    }

    /// The identity translation (the zero vector). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Translation6<T> {
        Translation6 {
            vector: Vector6 {
                x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero(), a: R::zero(), b: R::zero(),
            },
        }
    }

    /// The translation by `v`. Exact. Upstream: `from_vector` (deprecated upstream for `From`),
    /// also `v.into()`.
    #[inline(always)]
    fn from_vector(v: Vector6<T>) -> Translation6<T> {
        Translation6 { vector: v }
    }

    /// The inverse translation, by `-vector`. Exact; panics on overflow (`-MIN`). Upstream:
    /// `inverse`.
    #[inline(always)]
    fn inverse(self: Translation6<T>) -> Translation6<T> {
        let v = self.vector;
        Translation6 { vector: Vector6 { x: -v.x, y: -v.y, z: -v.z, w: -v.w, a: -v.a, b: -v.b } }
    }

    /// `self * p`: the point translated by `vector`. Exact; panics on overflow. Upstream:
    /// `transform_point` (`t * p`).
    #[inline(always)]
    fn transform_point(self: Translation6<T>, p: Point6<T>) -> Point6<T> {
        let v = self.vector;
        Point6 {
            x: p.x + v.x, y: p.y + v.y, z: p.z + v.z, w: p.w + v.w, a: p.a + v.a, b: p.b + v.b,
        }
    }

    /// `self⁻¹ * p`: the point translated by `-vector`, as one subtraction. Exact; panics on
    /// overflow. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Translation6<T>, p: Point6<T>) -> Point6<T> {
        let v = self.vector;
        Point6 {
            x: p.x - v.x, y: p.y - v.y, z: p.z - v.z, w: p.w - v.w, a: p.a - v.a, b: p.b - v.b,
        }
    }

    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Translation6<T>, other: Translation6<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
            && R::abs_diff_eq(a.y, b.y, ulps)
            && R::abs_diff_eq(a.z, b.z, ulps)
            && R::abs_diff_eq(a.w, b.w, ulps)
            && R::abs_diff_eq(a.a, b.a, ulps)
            && R::abs_diff_eq(a.b, b.b, ulps)
    }

    /// `true` when every component is `relative_eq` to the matching component of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Translation6<T>, other: Translation6<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.z, b.z, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.w, b.w, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.a, b.a, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.b, b.b, epsilon, max_relative)
    }

    /// `true` when every component is `ulps_eq` to the matching component of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Translation6<T>, other: Translation6<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.z, b.z, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.w, b.w, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.a, b.a, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.b, b.b, epsilon, max_ulps)
    }

    /// The same translation with every component converted by `Into<T, U>` (the identity for
    /// the single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Translation<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Translation6<T>) -> Translation6<U> {
        let v = self.vector;
        Translation6 {
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

/// `a * b`: the composition of two translations, the SUM of their vectors. Exact; panics on
/// overflow. Upstream: `Mul`.
pub impl Translation6Mul<T, +Add<T>, +Copy<T>, +Drop<T>> of Mul<Translation6<T>> {
    #[inline(always)]
    fn mul(lhs: Translation6<T>, rhs: Translation6<T>) -> Translation6<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation6 {
            vector: Vector6 {
                x: a.x + b.x, y: a.y + b.y, z: a.z + b.z, w: a.w + b.w, a: a.a + b.a, b: a.b + b.b,
            },
        }
    }
}

/// `v.into()`: the translation by `v`. Upstream: `From<Vector6> for Translation6`.
pub impl Translation6FromVector<T> of Into<Vector6<T>, Translation6<T>> {
    #[inline(always)]
    fn into(self: Vector6<T>) -> Translation6<T> {
        Translation6 { vector: self }
    }
}

/// `a / b = a * b⁻¹`: the translation by `a.vector - b.vector`. Exact; panics on overflow.
/// Upstream: `Div<Translation>`.
pub impl Translation6Div<T, +Sub<T>, +Copy<T>, +Drop<T>> of Div<Translation6<T>> {
    #[inline(always)]
    fn div(lhs: Translation6<T>, rhs: Translation6<T>) -> Translation6<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation6 {
            vector: Vector6 {
                x: a.x - b.x, y: a.y - b.y, z: a.z - b.z, w: a.w - b.w, a: a.a - b.a, b: a.b - b.b,
            },
        }
    }
}

/// `One::one()`: the identity translation; `is_one` tests for the zero vector exactly.
/// Upstream: `num::One for Translation`.
pub impl Translation6One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Translation6<T>> {
    #[inline(always)]
    fn one() -> Translation6<T> {
        Translation6 {
            vector: Vector6 {
                x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero(), a: R::zero(), b: R::zero(),
            },
        }
    }

    #[inline(always)]
    fn is_one(self: @Translation6<T>) -> bool {
        let v = *self.vector;
        v.x == R::zero()
            && v.y == R::zero()
            && v.z == R::zero()
            && v.w == R::zero()
            && v.a == R::zero()
            && v.b == R::zero()
    }

    #[inline(always)]
    fn is_non_one(self: @Translation6<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `p.into()`: the translation by the position vector of `p`. Upstream: `From<Point6> for
/// Translation6`.
pub impl Translation6FromPoint<T> of Into<Point6<T>, Translation6<T>> {
    #[inline(always)]
    fn into(self: Point6<T>) -> Translation6<T> {
        let Point6 { x, y, z, w, a, b } = self;
        Translation6 { vector: Vector6 { x, y, z, w, a, b } }
    }
}

/// `[x, y, z, w, a, b].into()`: the translation by that vector. Upstream: `From<[T; 6]>`.
pub impl Translation6FromArray<T> of Into<[T; 6], Translation6<T>> {
    #[inline(always)]
    fn into(self: [T; 6]) -> Translation6<T> {
        let [x, y, z, w, a, b] = self;
        Translation6 { vector: Vector6 { x, y, z, w, a, b } }
    }
}

/// The components of the vector as an array. Upstream: `Into<[T; 6]>`.
pub impl Translation6IntoArray<T> of Into<Translation6<T>, [T; 6]> {
    #[inline(always)]
    fn into(self: Translation6<T>) -> [T; 6] {
        let Vector6 { x, y, z, w, a, b } = self.vector;
        [x, y, z, w, a, b]
    }
}
