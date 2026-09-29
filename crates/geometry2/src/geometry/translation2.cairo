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

use nalgebra_core::internal::geometry::quaternion::ApproxEqTrait;
use nalgebra_types2::base::point2::Point2;
use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types2::geometry::rotation2::Rotation2;
use nalgebra_types2::geometry::translation2::Translation2;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::internal::geometry::translation2::Matrix3FromTranslation2KernelTrait;
use simba::scalar::Real;
use crate::geometry::isometry2::Isometry2;
use crate::geometry::isometry_matrix2::IsometryMatrix2;
use crate::geometry::similarity2::Similarity2;
use crate::geometry::unit_complex::UnitComplex;

/// Operations of `Translation2<T>` over a `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl Translation2Impl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>,
> of Translation2Trait<T> {
    /// The translation by `(x, y)`. Exact. Upstream: `Translation2::new`.
    #[inline(always)]
    fn new(x: T, y: T) -> Translation2<T> {
        Translation2 { vector: Vector2 { x, y } }
    }

    /// The identity translation (the zero vector). Exact. Upstream: `Translation2::identity`.
    #[inline(always)]
    fn identity() -> Translation2<T> {
        Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } }
    }

    /// The translation by `v`. Exact. Upstream: `Translation2::from(v)` (`From<Vector2>`), also
    /// available as `v.into()`.
    #[inline(always)]
    fn from_vector(v: Vector2<T>) -> Translation2<T> {
        Translation2 { vector: v }
    }

    /// The inverse translation, by `-vector`. Exact; panics on overflow (`-MIN`). Upstream:
    /// `inverse`.
    #[inline(always)]
    fn inverse(self: Translation2<T>) -> Translation2<T> {
        Translation2 { vector: Vector2 { x: -self.vector.x, y: -self.vector.y } }
    }

    /// `self = self.inverse()` in place (`-vector`; the by-value form is the cheapest, so the bits
    /// are those of `inverse`). Exact; panics on overflow (`-MIN`). Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: Translation2<T>) {
        self = Self::inverse(self);
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
        Matrix3FromTranslation2KernelTrait::to_homogeneous(self)
    }

    /// `true` when both components are within `ulps` smallest units (raw units for fixed point)
    /// of `other`'s; cannot overflow. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance
    /// being counted in ulp instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Translation2<T>, other: Translation2<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.vector.x, other.vector.x, ulps)
            && R::abs_diff_eq(self.vector.y, other.vector.y, ulps)
    }

    /// `true` when every component is `relative_eq` to the matching component of `other` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Translation2<T>, other: Translation2<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::relative_eq(a.x, b.x, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.y, b.y, epsilon, max_relative)
    }

    /// `true` when every component is `ulps_eq` to the matching component of `other` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Translation2<T>, other: Translation2<T>, epsilon: u64, max_ulps: u32) -> bool {
        let (a, b) = (self.vector, other.vector);
        ApproxEqTrait::ulps_eq(a.x, b.x, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.y, b.y, epsilon, max_ulps)
    }

    /// The same translation with every component converted by `Into<T, U>` (the identity for
    /// the single scalar `Fixed`). Upstream: `cast` (and `SubsetOf<Translation<U>>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Translation2<T>) -> Translation2<U> {
        let v = self.vector;
        Translation2 { vector: Vector2 { x: v.x.into(), y: v.y.into() } }
    }

    // --- P09a completion: heterogeneous operators (Cairo's operator traits are homogeneous) ----

    /// `self * iso`: the isometry of rotation `iso.rotation` and translation
    /// `self.vector + iso.translation.vector` (exact additions). Upstream: `Mul<Isometry2> for
    /// Translation2` (`t * iso`).
    #[inline(always)]
    fn mul_isometry(self: Translation2<T>, iso: Isometry2<T>) -> Isometry2<T> {
        Isometry2 { rotation: iso.rotation, translation: self * iso.translation }
    }

    /// `self * sim`: `sim` with `self.vector` added to its translation (exact additions; the
    /// rotation and the scaling unchanged). Upstream: `Mul<Similarity2> for Translation2` (`t *
    /// s`).
    #[inline(always)]
    fn mul_similarity(self: Translation2<T>, sim: Similarity2<T>) -> Similarity2<T> {
        Similarity2 { isometry: Self::mul_isometry(self, sim.isometry), scaling: sim.scaling }
    }

    /// `self * r`: the isometry of rotation `r` and translation `self` (no arithmetic).
    /// Upstream: `Mul<UnitComplex> for Translation2` (`t * r`).
    #[inline(always)]
    fn mul_unit_complex(self: Translation2<T>, r: UnitComplex<T>) -> Isometry2<T> {
        Isometry2 { rotation: r, translation: self }
    }

    /// `self * r`: the isometry of rotation `r` and translation `self` (no arithmetic). Upstream:
    /// `Mul<Rotation> for Translation` (output `IsometryMatrix2`).
    #[inline(always)]
    fn mul_rotation(self: Translation2<T>, r: Rotation2<T>) -> IsometryMatrix2<T> {
        IsometryMatrix2 { rotation: r, translation: self }
    }
}
