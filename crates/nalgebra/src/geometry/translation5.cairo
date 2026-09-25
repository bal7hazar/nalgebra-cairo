//! `Translation5`: a 5-dimensional translation (upstream `nalgebra::Translation5`, i.e.
//! `Translation<T, 5>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector5`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::Real;
use crate::base::matrix6::Matrix6;
use crate::base::vector5::Vector5;
use super::point5::Point5;
use super::quaternion::ApproxEqTrait;

/// A translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation5<T> {
    pub vector: Vector5<T>,
}

/// Operations of `Translation5<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation5Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Translation5Trait<T> {
    /// The translation by `(x, y, z, w, a)`. Exact. Upstream: `Translation5::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T, a: T) -> Translation5<T> {
        Translation5 { vector: Vector5 { x, y, z, w, a } }
    }

    /// The identity translation (the zero vector). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Translation5<T> {
        Translation5 {
            vector: Vector5 {
                x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero(), a: R::zero(),
            },
        }
    }

    /// The translation by `v`. Exact. Upstream: `from_vector` (deprecated upstream for `From`),
    /// also `v.into()`.
    #[inline(always)]
    fn from_vector(v: Vector5<T>) -> Translation5<T> {
        Translation5 { vector: v }
    }

    /// The inverse translation, by `-vector`. Exact; panics on overflow (`-MIN`). Upstream:
    /// `inverse`.
    #[inline(always)]
    fn inverse(self: Translation5<T>) -> Translation5<T> {
        let v = self.vector;
        Translation5 { vector: Vector5 { x: -v.x, y: -v.y, z: -v.z, w: -v.w, a: -v.a } }
    }

    /// `self = self.inverse()` in place (`-vector`; the by-value form is the cheapest, so the bits
    /// are those of `inverse`). Exact; panics on overflow (`-MIN`). Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: Translation5<T>) {
        self = Self::inverse(self);
    }

    /// `self * p`: the point translated by `vector`. Exact; panics on overflow. Upstream:
    /// `transform_point` (`t * p`).
    #[inline(always)]
    fn transform_point(self: Translation5<T>, p: Point5<T>) -> Point5<T> {
        let v = self.vector;
        Point5 { x: p.x + v.x, y: p.y + v.y, z: p.z + v.z, w: p.w + v.w, a: p.a + v.a }
    }

    /// `self⁻¹ * p`: the point translated by `-vector`, as one subtraction. Exact; panics on
    /// overflow. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Translation5<T>, p: Point5<T>) -> Point5<T> {
        let v = self.vector;
        Point5 { x: p.x - v.x, y: p.y - v.y, z: p.z - v.z, w: p.w - v.w, a: p.a - v.a }
    }

    /// The translation as a 6x6 homogeneous matrix: the identity with `vector` in the last
    /// column. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Translation5<T>) -> Matrix6<T> {
        Matrix6 {
            m11: R::one(),
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m51: R::zero(),
            m61: R::zero(),
            m12: R::zero(),
            m22: R::one(),
            m32: R::zero(),
            m42: R::zero(),
            m52: R::zero(),
            m62: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
            m43: R::zero(),
            m53: R::zero(),
            m63: R::zero(),
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: R::one(),
            m54: R::zero(),
            m64: R::zero(),
            m15: R::zero(),
            m25: R::zero(),
            m35: R::zero(),
            m45: R::zero(),
            m55: R::one(),
            m65: R::zero(),
            m16: self.vector.x,
            m26: self.vector.y,
            m36: self.vector.z,
            m46: self.vector.w,
            m56: self.vector.a,
            m66: R::one(),
        }
    }

    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Translation5<T>, other: Translation5<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
            && R::abs_diff_eq(a.y, b.y, ulps)
            && R::abs_diff_eq(a.z, b.z, ulps)
            && R::abs_diff_eq(a.w, b.w, ulps)
            && R::abs_diff_eq(a.a, b.a, ulps)
    }

    /// `true` when every component is `relative_eq` to the matching component of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Translation5<T>, other: Translation5<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.z, b.z, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.w, b.w, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.a, b.a, epsilon, max_relative)
    }

    /// `true` when every component is `ulps_eq` to the matching component of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Translation5<T>, other: Translation5<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.z, b.z, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.w, b.w, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.a, b.a, epsilon, max_ulps)
    }

    /// The same translation with every component converted by `Into<T, U>` (the identity for
    /// the single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Translation<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Translation5<T>) -> Translation5<U> {
        let v = self.vector;
        Translation5 {
            vector: Vector5 {
                x: v.x.into(), y: v.y.into(), z: v.z.into(), w: v.w.into(), a: v.a.into(),
            },
        }
    }
}

/// `a * b`: the composition of two translations, the SUM of their vectors. Exact; panics on
/// overflow. Upstream: `Mul`.
pub impl Translation5Mul<T, +Add<T>, +Copy<T>, +Drop<T>> of Mul<Translation5<T>> {
    #[inline(always)]
    fn mul(lhs: Translation5<T>, rhs: Translation5<T>) -> Translation5<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation5 {
            vector: Vector5 {
                x: a.x + b.x, y: a.y + b.y, z: a.z + b.z, w: a.w + b.w, a: a.a + b.a,
            },
        }
    }
}

/// `v.into()`: the translation by `v`. Upstream: `From<Vector5> for Translation5`.
pub impl Translation5FromVector<T> of Into<Vector5<T>, Translation5<T>> {
    #[inline(always)]
    fn into(self: Vector5<T>) -> Translation5<T> {
        Translation5 { vector: self }
    }
}

/// `a / b = a * b⁻¹`: the translation by `a.vector - b.vector`. Exact; panics on overflow.
/// Upstream: `Div<Translation>`.
pub impl Translation5Div<T, +Sub<T>, +Copy<T>, +Drop<T>> of Div<Translation5<T>> {
    #[inline(always)]
    fn div(lhs: Translation5<T>, rhs: Translation5<T>) -> Translation5<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation5 {
            vector: Vector5 {
                x: a.x - b.x, y: a.y - b.y, z: a.z - b.z, w: a.w - b.w, a: a.a - b.a,
            },
        }
    }
}

/// `a *= b`: `a = a * b` (component-wise sums, exact). Upstream: `MulAssign<Translation>`.
pub impl Translation5MulAssign<
    T, +Add<T>, +Copy<T>, +Drop<T>,
> of MulAssign<Translation5<T>, Translation5<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Translation5<T>, rhs: Translation5<T>) {
        self = self * rhs;
    }
}

/// `a /= b`: `a = a / b` (component-wise differences, exact). Upstream: `DivAssign<Translation>`.
pub impl Translation5DivAssign<
    T, +Sub<T>, +Copy<T>, +Drop<T>,
> of DivAssign<Translation5<T>, Translation5<T>> {
    #[inline(always)]
    fn div_assign(ref self: Translation5<T>, rhs: Translation5<T>) {
        self = self / rhs;
    }
}

/// `One::one()`: the identity translation; `is_one` tests for the zero vector exactly.
/// Upstream: `num::One for Translation`.
pub impl Translation5One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Translation5<T>> {
    #[inline(always)]
    fn one() -> Translation5<T> {
        Translation5 {
            vector: Vector5 {
                x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero(), a: R::zero(),
            },
        }
    }

    #[inline(always)]
    fn is_one(self: @Translation5<T>) -> bool {
        let v = *self.vector;
        v.x == R::zero()
            && v.y == R::zero()
            && v.z == R::zero()
            && v.w == R::zero()
            && v.a == R::zero()
    }

    #[inline(always)]
    fn is_non_one(self: @Translation5<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `p.into()`: the translation by the position vector of `p`. Upstream: `From<Point5> for
/// Translation5`.
pub impl Translation5FromPoint<T> of Into<Point5<T>, Translation5<T>> {
    #[inline(always)]
    fn into(self: Point5<T>) -> Translation5<T> {
        let Point5 { x, y, z, w, a } = self;
        Translation5 { vector: Vector5 { x, y, z, w, a } }
    }
}

/// `[x, y, z, w, a].into()`: the translation by that vector. Upstream: `From<[T; 5]>`.
pub impl Translation5FromArray<T> of Into<[T; 5], Translation5<T>> {
    #[inline(always)]
    fn into(self: [T; 5]) -> Translation5<T> {
        let [x, y, z, w, a] = self;
        Translation5 { vector: Vector5 { x, y, z, w, a } }
    }
}

/// The components of the vector as an array. Upstream: `Into<[T; 5]>`.
pub impl Translation5IntoArray<T> of Into<Translation5<T>, [T; 5]> {
    #[inline(always)]
    fn into(self: Translation5<T>) -> [T; 5] {
        let Vector5 { x, y, z, w, a } = self.vector;
        [x, y, z, w, a]
    }
}
