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
use simba::scalar::Real;
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;
use super::isometry3::Isometry3;
use super::isometry_matrix3::IsometryMatrix3;
use super::quaternion::{ApproxEqTrait, Quaternion};
use super::rotation3::Rotation3;
use super::similarity3::Similarity3;
use super::unit_quaternion::UnitQuaternion;


/// A 3D translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation3<T> {
    pub vector: Vector3<T>,
}

/// Operations of `Translation3<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation3Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
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

    /// `true` when every component is `relative_eq` to the matching component of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Translation3<T>, other: Translation3<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.z, b.z, epsilon, max_relative)
    }

    /// `true` when every component is `ulps_eq` to the matching component of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Translation3<T>, other: Translation3<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.z, b.z, epsilon, max_ulps)
    }

    /// The same translation with every component converted by `Into<T, U>` (the identity for
    /// the single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Translation<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Translation3<T>) -> Translation3<U> {
        let v = self.vector;
        Translation3 { vector: Vector3 { x: v.x.into(), y: v.y.into(), z: v.z.into() } }
    }

    // --- P09a completion: heterogeneous operators (Cairo's operator traits are homogeneous) ----

    /// `self * iso`: the isometry of rotation `iso.rotation` and translation
    /// `self.vector + iso.translation.vector` (exact additions). Upstream: `Mul<Isometry3> for
    /// Translation3` (`t * iso`).
    #[inline(always)]
    fn mul_isometry(self: Translation3<T>, iso: Isometry3<T>) -> Isometry3<T> {
        Isometry3 { rotation: iso.rotation, translation: self * iso.translation }
    }

    /// `self * sim`: `sim` with `self.vector` added to its translation (exact additions; the
    /// rotation and the scaling unchanged). Upstream: `Mul<Similarity3> for Translation3` (`t *
    /// s`).
    #[inline(always)]
    fn mul_similarity(self: Translation3<T>, sim: Similarity3<T>) -> Similarity3<T> {
        Similarity3 { isometry: Self::mul_isometry(self, sim.isometry), scaling: sim.scaling }
    }

    /// `self * r`: the isometry of rotation `r` and translation `self` (no arithmetic).
    /// Upstream: `Mul<UnitQuaternion> for Translation3` (`t * r`).
    #[inline(always)]
    fn mul_unit_quaternion(self: Translation3<T>, r: UnitQuaternion<T>) -> Isometry3<T> {
        Isometry3 { rotation: r, translation: self }
    }

    /// `self * r`: the isometry of rotation `r` and translation `self` (no arithmetic). Upstream:
    /// `Mul<Rotation> for Translation` (output `IsometryMatrix3`).
    #[inline(always)]
    fn mul_rotation(self: Translation3<T>, r: Rotation3<T>) -> IsometryMatrix3<T> {
        IsometryMatrix3 { rotation: r, translation: self }
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

/// `a / b = a * b⁻¹`: the translation by `a.vector - b.vector`. Exact; panics on overflow.
/// Upstream: `Div<Translation>`.
pub impl Translation3Div<T, +Sub<T>, +Copy<T>, +Drop<T>> of Div<Translation3<T>> {
    #[inline(always)]
    fn div(lhs: Translation3<T>, rhs: Translation3<T>) -> Translation3<T> {
        let (a, b) = (lhs.vector, rhs.vector);
        Translation3 { vector: Vector3 { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z } }
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

/// `t.into()`: the similarity of translation `t`, identity rotation and scaling 1. Upstream:
/// `SubsetOf<Similarity3> for Translation3` (`nalgebra::convert(t)`).
pub impl Similarity3FromTranslation3<
    T, impl R: Real<T>, +Drop<T>,
> of Into<Translation3<T>, Similarity3<T>> {
    #[inline(always)]
    fn into(self: Translation3<T>) -> Similarity3<T> {
        Similarity3 {
            isometry: Isometry3 {
                rotation: UnitQuaternion {
                    quaternion: Quaternion {
                        i: R::zero(), j: R::zero(), k: R::zero(), w: R::one(),
                    },
                },
                translation: self,
            },
            scaling: R::one(),
        }
    }
}
