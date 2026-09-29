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
use crate::geometry::point1::Point1;

/// A translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation1<T> {
    pub vector: Matrix1<T>,
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
