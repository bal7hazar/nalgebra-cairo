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

use simba::scalar::Real;
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;


/// A 3D translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation3<T> {
    pub vector: Vector3<T>,
}

/// Operations of `Translation3<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation3Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Neg<T>,
> of Translation3Trait<T> {
    /// The translation by `(x, y, z)`. Exact. Upstream: `Translation3::new`.
    #[inline(always)]
    fn new(x: T, y: T, z: T) -> Translation3<T> {
        Translation3 { vector: Vector3 { x, y, z } }
    }

    /// The identity translation (the zero vector). Exact. Upstream: `Translation3::identity`.
    #[inline(always)]
    fn identity() -> Translation3<T> {
        Translation3 { vector: Vector3 { x: R::zero(), y: R::zero(), z: R::zero() } }
    }

    /// The translation by `v`. Exact. Upstream: `Translation3::from(v)` (`From<Vector3>`), also
    /// available as `v.into()`.
    #[inline(always)]
    fn from_vector(v: Vector3<T>) -> Translation3<T> {
        Translation3 { vector: v }
    }

    /// The inverse translation, by `-vector`. Exact; panics on overflow (`-MIN`). Upstream:
    /// `inverse`.
    #[inline(always)]
    fn inverse(self: Translation3<T>) -> Translation3<T> {
        Translation3 { vector: Vector3 { x: -self.vector.x, y: -self.vector.y, z: -self.vector.z } }
    }

    /// `self * p`: the point translated by `vector`. Exact; panics on overflow. Upstream:
    /// `transform_point` (`t * p`).
    #[inline(always)]
    fn transform_point(self: Translation3<T>, p: Point3<T>) -> Point3<T> {
        Point3 { x: p.x + self.vector.x, y: p.y + self.vector.y, z: p.z + self.vector.z }
    }

    /// `self⁻¹ * p`: the point translated by `-vector`, as one subtraction (the inverse
    /// translation is never formed, so `-MIN` cannot overflow here). Exact; panics on overflow.
    /// Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Translation3<T>, p: Point3<T>) -> Point3<T> {
        Point3 { x: p.x - self.vector.x, y: p.y - self.vector.y, z: p.z - self.vector.z }
    }

    /// The translation as a 4x4 homogeneous matrix: the identity with `vector` in the last
    /// column. Exact. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Translation3<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::one(),
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m12: R::zero(),
            m22: R::one(),
            m32: R::zero(),
            m42: R::zero(),
            m13: R::zero(),
            m23: R::zero(),
            m33: R::one(),
            m43: R::zero(),
            m14: self.vector.x,
            m24: self.vector.y,
            m34: self.vector.z,
            m44: R::one(),
        }
    }

    /// `true` when the three components are within `ulps` smallest units (raw units for fixed
    /// point) of `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the
    /// tolerance being counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Translation3<T>, other: Translation3<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.vector.x, other.vector.x, ulps)
            && R::abs_diff_eq(self.vector.y, other.vector.y, ulps)
            && R::abs_diff_eq(self.vector.z, other.vector.z, ulps)
    }
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
