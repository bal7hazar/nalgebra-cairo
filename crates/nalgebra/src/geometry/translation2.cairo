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

use simba::scalar::Real;
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

/// A 2D translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation2<T> {
    pub vector: Vector2<T>,
}

/// Operations of `Translation2<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation2Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Neg<T>,
> of Translation2Trait<T> {
    /// The translation by `(x, y)`. Exact. Upstream: `Translation2::new`.
    #[inline(always)]
    fn new(x: T, y: T) -> Translation2<T> {
        Translation2 { vector: Vector2 { x, y } }
    }

    /// The identity translation (the zero vector). Exact. Upstream: `Translation2::identity`.
    #[inline(always)]
    fn identity() -> Translation2<T> {
        Translation2 { vector: Vector2 { x: R::ZERO, y: R::ZERO } }
    }

    /// The translation by `v`. Exact. Upstream: `Translation2::from(v)` (`From<Vector2>`), also
    /// available as `v.into()`.
    #[inline(always)]
    fn from_vector(v: Vector2<T>) -> Translation2<T> {
        Translation2 { vector: v }
    }

    /// The translation vector (a copy: everything is by value here). Upstream: the `vector`
    /// field.
    #[inline(always)]
    fn vector(self: Translation2<T>) -> Vector2<T> {
        self.vector
    }

    /// The inverse translation, by `-vector`. Exact; panics on overflow (`-MIN`). Upstream:
    /// `inverse`.
    #[inline(always)]
    fn inverse(self: Translation2<T>) -> Translation2<T> {
        Translation2 { vector: Vector2 { x: -self.vector.x, y: -self.vector.y } }
    }

    /// `self * p`: the point translated by `vector`. Exact; panics on overflow. Upstream:
    /// `transform_point` (`t * p`).
    #[inline(always)]
    fn transform_point(self: Translation2<T>, p: Point2<T>) -> Point2<T> {
        Point2 { x: p.x + self.vector.x, y: p.y + self.vector.y }
    }

    /// `self⁻¹ * p`: the point translated by `-vector`, as one subtraction (the inverse
    /// translation is never formed, so `-MIN` cannot overflow here). Exact; panics on overflow.
    /// Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Translation2<T>, p: Point2<T>) -> Point2<T> {
        Point2 { x: p.x - self.vector.x, y: p.y - self.vector.y }
    }

    /// The translation as a 3x3 homogeneous matrix: the identity with `vector` in the last
    /// column. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Translation2<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::ONE,
            m21: R::ZERO,
            m31: R::ZERO,
            m12: R::ZERO,
            m22: R::ONE,
            m32: R::ZERO,
            m13: self.vector.x,
            m23: self.vector.y,
            m33: R::ONE,
        }
    }

    /// `true` when both components are within `ulps` smallest units (raw units for fixed point)
    /// of `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance
    /// being counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Translation2<T>, other: Translation2<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.vector.x, other.vector.x, ulps)
            && R::abs_diff_eq(self.vector.y, other.vector.y, ulps)
    }
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

/// `t.into()`: the translation vector. Upstream: the `vector` field.
pub impl Translation2IntoVector<T> of Into<Translation2<T>, Vector2<T>> {
    #[inline(always)]
    fn into(self: Translation2<T>) -> Vector2<T> {
        self.vector
    }
}
