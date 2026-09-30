//! In `nalgebra_types5`: the `struct` `Translation5`; its 9 impls, among them `Translation5Mul`,
//! `Translation5FromVector`, `Translation5Div` and 6 more. This module is split over packages; the
//! other parts are in `nalgebra_geometry5`.
//!
//! `Translation5`: a 5-dimensional translation (upstream `nalgebra::Translation5`, i.e.
//! `Translation<T, 5>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector5`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::Real;
use crate::base::vector5::Vector5;
use crate::geometry::point5::Point5;

/// A translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation5<T> {
    pub vector: Vector5<T>,
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
