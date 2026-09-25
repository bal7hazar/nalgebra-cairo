//! `Translation1`: a 1-dimensional translation (upstream `nalgebra::Translation1`, i.e.
//! `Translation<T, 1>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Matrix1`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::Real;
use crate::base::matrix1::Matrix1;
use crate::base::matrix2::Matrix2;
use super::point1::Point1;
use super::quaternion::ApproxEqTrait;

/// A translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation1<T> {
    pub vector: Matrix1<T>,
}

/// Operations of `Translation1<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation1Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Translation1Trait<T> {
    /// The translation by `(x)`. Exact. Upstream: `Translation1::new`.
    #[inline(always)]
    fn new(x: T) -> Translation1<T> {
        Translation1 { vector: Matrix1 { x } }
    }

    /// The identity translation (the zero vector). Exact. Upstream: `identity`.
    #[inline(always)]
    fn identity() -> Translation1<T> {
        Translation1 { vector: Matrix1 { x: R::zero() } }
    }

    /// The translation by `v`. Exact. Upstream: `from_vector` (deprecated upstream for `From`),
    /// also `v.into()`.
    #[inline(always)]
    fn from_vector(v: Matrix1<T>) -> Translation1<T> {
        Translation1 { vector: v }
    }

    /// The inverse translation, by `-vector`. Exact; panics on overflow (`-MIN`). Upstream:
    /// `inverse`.
    #[inline(always)]
    fn inverse(self: Translation1<T>) -> Translation1<T> {
        let v = self.vector;
        Translation1 { vector: Matrix1 { x: -v.x } }
    }

    /// `self = self.inverse()` in place (`-vector`; the by-value form is the cheapest, so the bits
    /// are those of `inverse`). Exact; panics on overflow (`-MIN`). Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: Translation1<T>) {
        self = Self::inverse(self);
    }

    /// `self * p`: the point translated by `vector`. Exact; panics on overflow. Upstream:
    /// `transform_point` (`t * p`).
    #[inline(always)]
    fn transform_point(self: Translation1<T>, p: Point1<T>) -> Point1<T> {
        let v = self.vector;
        Point1 { x: p.x + v.x }
    }

    /// `self⁻¹ * p`: the point translated by `-vector`, as one subtraction. Exact; panics on
    /// overflow. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Translation1<T>, p: Point1<T>) -> Point1<T> {
        let v = self.vector;
        Point1 { x: p.x - v.x }
    }

    /// The translation as a 2x2 homogeneous matrix: the identity with `vector` in the last
    /// column. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Translation1<T>) -> Matrix2<T> {
        Matrix2 { m11: R::one(), m21: R::zero(), m12: self.vector.x, m22: R::one() }
    }

    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Translation1<T>, other: Translation1<T>, ulps: u64) -> bool {
        let (a, b) = (self.vector, other.vector);
        R::abs_diff_eq(a.x, b.x, ulps)
    }

    /// `true` when every component is `relative_eq` to the matching component of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Translation1<T>, other: Translation1<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
    }

    /// `true` when every component is `ulps_eq` to the matching component of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Translation1<T>, other: Translation1<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
    }

    /// The same translation with every component converted by `Into<T, U>` (the identity for
    /// the single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Translation<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Translation1<T>) -> Translation1<U> {
        let v = self.vector;
        Translation1 { vector: Matrix1 { x: v.x.into() } }
    }
}

/// `a * b`: the composition of two translations, the SUM of their vectors. Exact; panics on
/// overflow. Upstream: `Mul`.
pub impl Translation1Mul<T, +Add<T>, +Copy<T>, +Drop<T>> of Mul<Translation1<T>> {
    #[inline(always)]
    fn mul(lhs: Translation1<T>, rhs: Translation1<T>) -> Translation1<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation1 { vector: Matrix1 { x: a.x + b.x } }
    }
}

/// `v.into()`: the translation by `v`. Upstream: `From<Matrix1> for Translation1`.
pub impl Translation1FromVector<T> of Into<Matrix1<T>, Translation1<T>> {
    #[inline(always)]
    fn into(self: Matrix1<T>) -> Translation1<T> {
        Translation1 { vector: self }
    }
}

/// `a / b = a * b⁻¹`: the translation by `a.vector - b.vector`. Exact; panics on overflow.
/// Upstream: `Div<Translation>`.
pub impl Translation1Div<T, +Sub<T>, +Copy<T>, +Drop<T>> of Div<Translation1<T>> {
    #[inline(always)]
    fn div(lhs: Translation1<T>, rhs: Translation1<T>) -> Translation1<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation1 { vector: Matrix1 { x: a.x - b.x } }
    }
}

/// `a *= b`: `a = a * b` (component-wise sums, exact). Upstream: `MulAssign<Translation>`.
pub impl Translation1MulAssign<
    T, +Add<T>, +Copy<T>, +Drop<T>,
> of MulAssign<Translation1<T>, Translation1<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Translation1<T>, rhs: Translation1<T>) {
        self = self * rhs;
    }
}

/// `a /= b`: `a = a / b` (component-wise differences, exact). Upstream: `DivAssign<Translation>`.
pub impl Translation1DivAssign<
    T, +Sub<T>, +Copy<T>, +Drop<T>,
> of DivAssign<Translation1<T>, Translation1<T>> {
    #[inline(always)]
    fn div_assign(ref self: Translation1<T>, rhs: Translation1<T>) {
        self = self / rhs;
    }
}

/// `One::one()`: the identity translation; `is_one` tests for the zero vector exactly.
/// Upstream: `num::One for Translation`.
pub impl Translation1One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Translation1<T>> {
    #[inline(always)]
    fn one() -> Translation1<T> {
        Translation1 { vector: Matrix1 { x: R::zero() } }
    }

    #[inline(always)]
    fn is_one(self: @Translation1<T>) -> bool {
        let v = *self.vector;
        v.x == R::zero()
    }

    #[inline(always)]
    fn is_non_one(self: @Translation1<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `p.into()`: the translation by the position vector of `p`. Upstream: `From<Point1> for
/// Translation1`.
pub impl Translation1FromPoint<T> of Into<Point1<T>, Translation1<T>> {
    #[inline(always)]
    fn into(self: Point1<T>) -> Translation1<T> {
        let Point1 { x } = self;
        Translation1 { vector: Matrix1 { x } }
    }
}

/// `[x].into()`: the translation by that vector. Upstream: `From<[T; 1]>`.
pub impl Translation1FromArray<T> of Into<[T; 1], Translation1<T>> {
    #[inline(always)]
    fn into(self: [T; 1]) -> Translation1<T> {
        let [x] = self;
        Translation1 { vector: Matrix1 { x } }
    }
}

/// The components of the vector as an array. Upstream: `Into<[T; 1]>`.
pub impl Translation1IntoArray<T> of Into<Translation1<T>, [T; 1]> {
    #[inline(always)]
    fn into(self: Translation1<T>) -> [T; 1] {
        let Matrix1 { x } = self.vector;
        [x]
    }
}
