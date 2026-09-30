//! In `nalgebra_types4`: the `struct` `Translation4`; its 9 impls, among them `Translation4Mul`,
//! `Translation4FromVector`, `Translation4Div` and 6 more. This module is split over packages; the
//! other parts are in `nalgebra_geometry4`.
//!
//! `Translation4`: a 4-dimensional translation (upstream `nalgebra::Translation4`, i.e.
//! `Translation<T, 4>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector4`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::Real;
use crate::base::vector4::Vector4;
use crate::geometry::point4::Point4;

/// A translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation4<T> {
    pub vector: Vector4<T>,
}

/// `a * b`: the composition of two translations, the SUM of their vectors. Exact; panics on
/// overflow. Upstream: `Mul`.
pub impl Translation4Mul<T, +Add<T>, +Copy<T>, +Drop<T>> of Mul<Translation4<T>> {
    #[inline(always)]
    fn mul(lhs: Translation4<T>, rhs: Translation4<T>) -> Translation4<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation4 { vector: Vector4 { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z, w: a.w + b.w } }
    }
}

/// `v.into()`: the translation by `v`. Upstream: `From<Vector4> for Translation4`.
pub impl Translation4FromVector<T> of Into<Vector4<T>, Translation4<T>> {
    #[inline(always)]
    fn into(self: Vector4<T>) -> Translation4<T> {
        Translation4 { vector: self }
    }
}

/// `a / b = a * b⁻¹`: the translation by `a.vector - b.vector`. Exact; panics on overflow.
/// Upstream: `Div<Translation>`.
pub impl Translation4Div<T, +Sub<T>, +Copy<T>, +Drop<T>> of Div<Translation4<T>> {
    #[inline(always)]
    fn div(lhs: Translation4<T>, rhs: Translation4<T>) -> Translation4<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation4 { vector: Vector4 { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z, w: a.w - b.w } }
    }
}

/// `a *= b`: `a = a * b` (component-wise sums, exact). Upstream: `MulAssign<Translation>`.
pub impl Translation4MulAssign<
    T, +Add<T>, +Copy<T>, +Drop<T>,
> of MulAssign<Translation4<T>, Translation4<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Translation4<T>, rhs: Translation4<T>) {
        self = self * rhs;
    }
}

/// `a /= b`: `a = a / b` (component-wise differences, exact). Upstream: `DivAssign<Translation>`.
pub impl Translation4DivAssign<
    T, +Sub<T>, +Copy<T>, +Drop<T>,
> of DivAssign<Translation4<T>, Translation4<T>> {
    #[inline(always)]
    fn div_assign(ref self: Translation4<T>, rhs: Translation4<T>) {
        self = self / rhs;
    }
}

/// `One::one()`: the identity translation; `is_one` tests for the zero vector exactly.
/// Upstream: `num::One for Translation`.
pub impl Translation4One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Translation4<T>> {
    #[inline(always)]
    fn one() -> Translation4<T> {
        Translation4 { vector: Vector4 { x: R::zero(), y: R::zero(), z: R::zero(), w: R::zero() } }
    }

    #[inline(always)]
    fn is_one(self: @Translation4<T>) -> bool {
        let v = *self.vector;
        v.x == R::zero() && v.y == R::zero() && v.z == R::zero() && v.w == R::zero()
    }

    #[inline(always)]
    fn is_non_one(self: @Translation4<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `p.into()`: the translation by the position vector of `p`. Upstream: `From<Point4> for
/// Translation4`.
pub impl Translation4FromPoint<T> of Into<Point4<T>, Translation4<T>> {
    #[inline(always)]
    fn into(self: Point4<T>) -> Translation4<T> {
        let Point4 { x, y, z, w } = self;
        Translation4 { vector: Vector4 { x, y, z, w } }
    }
}

/// `[x, y, z, w].into()`: the translation by that vector. Upstream: `From<[T; 4]>`.
pub impl Translation4FromArray<T> of Into<[T; 4], Translation4<T>> {
    #[inline(always)]
    fn into(self: [T; 4]) -> Translation4<T> {
        let [x, y, z, w] = self;
        Translation4 { vector: Vector4 { x, y, z, w } }
    }
}

/// The components of the vector as an array. Upstream: `Into<[T; 4]>`.
pub impl Translation4IntoArray<T> of Into<Translation4<T>, [T; 4]> {
    #[inline(always)]
    fn into(self: Translation4<T>) -> [T; 4] {
        let Vector4 { x, y, z, w } = self.vector;
        [x, y, z, w]
    }
}
