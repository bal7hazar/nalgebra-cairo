//! In `nalgebra_types2`: the `struct` `Translation2`; its 9 impls, among them `Translation2Mul`,
//! `Translation2FromVector`, `Translation2Div` and 6 more. This module is split over packages; the
//! other parts are in `nalgebra_geometry2`.
//!
//! `Translation2`: a 2D translation (upstream `nalgebra::Translation2`, which is
//! `Translation<T, 2>`).
//!
//! - `Translation2Trait` / `Translation2Impl`: construction, accessors, composition, point
//!   transforms, the homogeneous matrix and comparison — everything a translation can do is
//!   algebraic, so the whole type lives over `simba::scalar::Real` and needs no `*AngleTrait`;
//! - `a * b` (composition) and the conversions from / to `Vector2<T>`: their impls live in this
//!   module, where the compiler finds them without any import.
//!
//! A translation acts on POINTS only: a `Vector2` is a displacement, which a translation leaves
//! unchanged (upstream has no `Translation::transform_vector` either). Every operation of this
//! type is an exact addition or negation: nothing here rounds, and the oracle tolerance of the
//! whole `translation` suite is 0.
//!
//! Numeric contract (AGENTS.md): additions and negations panic instead of wrapping; no sum of
//! products is formed here, so no fused kernel is needed.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::Real;
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;

/// A 2D translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation2<T> {
    pub vector: Vector2<T>,
}

/// `a * b`: the composition of two translations, the SUM of their vectors (translations commute).
/// Exact; panics on overflow. Upstream: `Mul`.
pub impl Translation2Mul<T, +Add<T>, +Copy<T>, +Drop<T>> of Mul<Translation2<T>> {
    #[inline(always)]
    fn mul(lhs: Translation2<T>, rhs: Translation2<T>) -> Translation2<T> {
        Translation2 {
            vector: Vector2 { x: lhs.vector.x + rhs.vector.x, y: lhs.vector.y + rhs.vector.y },
        }
    }
}

/// `v.into()`: the translation by `v`. Upstream: `From<Vector2> for Translation2`.
pub impl Translation2FromVector<T> of Into<Vector2<T>, Translation2<T>> {
    #[inline(always)]
    fn into(self: Vector2<T>) -> Translation2<T> {
        Translation2 { vector: self }
    }
}

/// `a / b = a * b⁻¹`: the translation by `a.vector - b.vector`. Exact; panics on overflow.
/// Upstream: `Div<Translation>`.
pub impl Translation2Div<T, +Sub<T>, +Copy<T>, +Drop<T>> of Div<Translation2<T>> {
    #[inline(always)]
    fn div(lhs: Translation2<T>, rhs: Translation2<T>) -> Translation2<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation2 { vector: Vector2 { x: a.x - b.x, y: a.y - b.y } }
    }
}

/// `a *= b`: `a = a * b` (component-wise sums, exact). Upstream: `MulAssign<Translation>`.
pub impl Translation2MulAssign<
    T, +Add<T>, +Copy<T>, +Drop<T>,
> of MulAssign<Translation2<T>, Translation2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Translation2<T>, rhs: Translation2<T>) {
        self = self * rhs;
    }
}

/// `a /= b`: `a = a / b` (component-wise differences, exact). Upstream: `DivAssign<Translation>`.
pub impl Translation2DivAssign<
    T, +Sub<T>, +Copy<T>, +Drop<T>,
> of DivAssign<Translation2<T>, Translation2<T>> {
    #[inline(always)]
    fn div_assign(ref self: Translation2<T>, rhs: Translation2<T>) {
        self = self / rhs;
    }
}

/// `One::one()`: the identity translation; `is_one` tests for the zero vector exactly.
/// Upstream: `num::One for Translation`.
pub impl Translation2One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Translation2<T>> {
    #[inline(always)]
    fn one() -> Translation2<T> {
        Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } }
    }

    #[inline(always)]
    fn is_one(self: @Translation2<T>) -> bool {
        let v = *self.vector;
        v.x == R::zero() && v.y == R::zero()
    }

    #[inline(always)]
    fn is_non_one(self: @Translation2<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `p.into()`: the translation by the position vector of `p`. Upstream: `From<Point2> for
/// Translation2`.
pub impl Translation2FromPoint<T> of Into<Point2<T>, Translation2<T>> {
    #[inline(always)]
    fn into(self: Point2<T>) -> Translation2<T> {
        let Point2 { x, y } = self;
        Translation2 { vector: Vector2 { x, y } }
    }
}

/// `[x, y].into()`: the translation by that vector. Upstream: `From<[T; 2]>`.
pub impl Translation2FromArray<T> of Into<[T; 2], Translation2<T>> {
    #[inline(always)]
    fn into(self: [T; 2]) -> Translation2<T> {
        let [x, y] = self;
        Translation2 { vector: Vector2 { x, y } }
    }
}

/// The components of the vector as an array. Upstream: `Into<[T; 2]>`.
pub impl Translation2IntoArray<T> of Into<Translation2<T>, [T; 2]> {
    #[inline(always)]
    fn into(self: Translation2<T>) -> [T; 2] {
        let Vector2 { x, y } = self.vector;
        [x, y]
    }
}
