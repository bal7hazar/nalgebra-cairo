//! `Translation3`: a 3D translation (upstream `nalgebra::Translation3`, which is
//! `Translation<T, 3>`).
//!
//! - `Translation3Trait` / `Translation3Impl`: construction, accessors, composition, point
//!   transforms, the homogeneous matrix and comparison — everything a translation can do is
//!   algebraic, so the whole type lives over `simba::scalar::Real` and needs no `*AngleTrait`;
//! - `a * b` (composition) and the conversions from / to `Vector3<T>`: their impls live in this
//!   module, where the compiler finds them without any import.
//!
//! A translation acts on POINTS only: a `Vector3` is a displacement, which a translation leaves
//! unchanged (upstream has no `Translation::transform_vector` either). Every operation of this
//! type is an exact addition or negation: nothing here rounds, and the oracle tolerance of the
//! whole `translation` suite is 0.
//!
//! Numeric contract (AGENTS.md): additions and negations panic instead of wrapping; no sum of
//! products is formed here, so no fused kernel is needed.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::Real;
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;

/// A 3D translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation3<T> {
    pub vector: Vector3<T>,
}

/// `a * b`: the composition of two translations, the SUM of their vectors (translations commute).
/// Exact; panics on overflow. Upstream: `Mul`.
pub impl Translation3Mul<T, +Add<T>, +Copy<T>, +Drop<T>> of Mul<Translation3<T>> {
    #[inline(always)]
    fn mul(lhs: Translation3<T>, rhs: Translation3<T>) -> Translation3<T> {
        Translation3 {
            vector: Vector3 {
                x: lhs.vector.x + rhs.vector.x,
                y: lhs.vector.y + rhs.vector.y,
                z: lhs.vector.z + rhs.vector.z,
            },
        }
    }
}

/// `v.into()`: the translation by `v`. Upstream: `From<Vector3> for Translation3`.
pub impl Translation3FromVector<T> of Into<Vector3<T>, Translation3<T>> {
    #[inline(always)]
    fn into(self: Vector3<T>) -> Translation3<T> {
        Translation3 { vector: self }
    }
}

/// `a / b = a * b⁻¹`: the translation by `a.vector - b.vector`. Exact; panics on overflow.
/// Upstream: `Div<Translation>`.
pub impl Translation3Div<T, +Sub<T>, +Copy<T>, +Drop<T>> of Div<Translation3<T>> {
    #[inline(always)]
    fn div(lhs: Translation3<T>, rhs: Translation3<T>) -> Translation3<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation3 { vector: Vector3 { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z } }
    }
}

/// `a *= b`: `a = a * b` (component-wise sums, exact). Upstream: `MulAssign<Translation>`.
pub impl Translation3MulAssign<
    T, +Add<T>, +Copy<T>, +Drop<T>,
> of MulAssign<Translation3<T>, Translation3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Translation3<T>, rhs: Translation3<T>) {
        self = self * rhs;
    }
}

/// `a /= b`: `a = a / b` (component-wise differences, exact). Upstream: `DivAssign<Translation>`.
pub impl Translation3DivAssign<
    T, +Sub<T>, +Copy<T>, +Drop<T>,
> of DivAssign<Translation3<T>, Translation3<T>> {
    #[inline(always)]
    fn div_assign(ref self: Translation3<T>, rhs: Translation3<T>) {
        self = self / rhs;
    }
}

/// `One::one()`: the identity translation; `is_one` tests for the zero vector exactly.
/// Upstream: `num::One for Translation`.
pub impl Translation3One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Translation3<T>> {
    #[inline(always)]
    fn one() -> Translation3<T> {
        Translation3 { vector: Vector3 { x: R::zero(), y: R::zero(), z: R::zero() } }
    }

    #[inline(always)]
    fn is_one(self: @Translation3<T>) -> bool {
        let v = *self.vector;
        v.x == R::zero() && v.y == R::zero() && v.z == R::zero()
    }

    #[inline(always)]
    fn is_non_one(self: @Translation3<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `p.into()`: the translation by the position vector of `p`. Upstream: `From<Point3> for
/// Translation3`.
pub impl Translation3FromPoint<T> of Into<Point3<T>, Translation3<T>> {
    #[inline(always)]
    fn into(self: Point3<T>) -> Translation3<T> {
        let Point3 { x, y, z } = self;
        Translation3 { vector: Vector3 { x, y, z } }
    }
}

/// `[x, y, z].into()`: the translation by that vector. Upstream: `From<[T; 3]>`.
pub impl Translation3FromArray<T> of Into<[T; 3], Translation3<T>> {
    #[inline(always)]
    fn into(self: [T; 3]) -> Translation3<T> {
        let [x, y, z] = self;
        Translation3 { vector: Vector3 { x, y, z } }
    }
}

/// The components of the vector as an array. Upstream: `Into<[T; 3]>`.
pub impl Translation3IntoArray<T> of Into<Translation3<T>, [T; 3]> {
    #[inline(always)]
    fn into(self: Translation3<T>) -> [T; 3] {
        let Vector3 { x, y, z } = self.vector;
        [x, y, z]
    }
}
