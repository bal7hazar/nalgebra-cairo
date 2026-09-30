//! `Translation6`: a 6-dimensional translation (upstream `nalgebra::Translation6`, i.e.
//! `Translation<T, 6>`), WP 8.4-P09a.
//!
//! The same API as `Translation2` / `Translation3` with the WP 8.4-P09a completion, written from
//! one template for the sizes 1, 4, 5 and 6 (the vector is the matching shape `Vector6`). Every
//! operation is an exact addition, subtraction or negation: nothing rounds; overflow panics.
//!
//! There is no `to_homogeneous`: the homogeneous matrix of a 6D translation is 7x7, and the static
//! shapes stop at 6 (DESIGN D4). Out of scope for 0.1.0 by owner ruling (issue #41).

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::Real;
use crate::base::vector6::Vector6;
use crate::geometry::point6::Point6;

/// A translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation6<T> {
    pub vector: Vector6<T>,
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

/// `a *= b`: `a = a * b` (component-wise sums, exact). Upstream: `MulAssign<Translation>`.
pub impl Translation6MulAssign<
    T, +Add<T>, +Copy<T>, +Drop<T>,
> of MulAssign<Translation6<T>, Translation6<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Translation6<T>, rhs: Translation6<T>) {
        self = self * rhs;
    }
}

/// `a /= b`: `a = a / b` (component-wise differences, exact). Upstream: `DivAssign<Translation>`.
pub impl Translation6DivAssign<
    T, +Sub<T>, +Copy<T>, +Drop<T>,
> of DivAssign<Translation6<T>, Translation6<T>> {
    #[inline(always)]
    fn div_assign(ref self: Translation6<T>, rhs: Translation6<T>) {
        self = self / rhs;
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
