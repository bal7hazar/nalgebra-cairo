//! In `nalgebra_geometry3`: the impl `Translation3Impl`. This module is split over packages; the
//! other parts are in `nalgebra_types3`.
//!
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

use nalgebra_core::internal::geometry::quaternion::ApproxEqTrait;
use nalgebra_types3::base::point3::Point3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::geometry::rotation3::Rotation3;
use nalgebra_types3::geometry::translation3::Translation3;
use nalgebra_types4::base::matrix4::Matrix4;
use nalgebra_types4::internal::geometry::translation3::Matrix4FromTranslation3KernelTrait;
use simba::scalar::Real;
use crate::geometry::isometry3::Isometry3;
use crate::geometry::isometry_matrix3::IsometryMatrix3;
use crate::geometry::similarity3::Similarity3;
use crate::geometry::unit_quaternion::UnitQuaternion;

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

    /// `self = self.inverse()` in place (`-vector`; the by-value form is the cheapest, so the bits
    /// are those of `inverse`). Exact; panics on overflow (`-MIN`). Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: Translation3<T>) {
        self = Self::inverse(self);
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
        Matrix4FromTranslation3KernelTrait::to_homogeneous(self)
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
