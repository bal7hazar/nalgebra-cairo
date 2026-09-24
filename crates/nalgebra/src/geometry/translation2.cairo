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
use simba::scalar::Real;
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;
use super::isometry2::Isometry2;
use super::quaternion::ApproxEqTrait;
use super::similarity2::Similarity2;
use super::unit_complex::UnitComplex;


/// A 2D translation by `vector`. The field name is upstream's (`Translation { vector }`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Translation2<T> {
    pub vector: Vector2<T>,
}

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
            m11: R::one(),
            m21: R::zero(),
            m31: R::zero(),
            m12: R::zero(),
            m22: R::one(),
            m32: R::zero(),
            m13: self.vector.x,
            m23: self.vector.y,
            m33: R::one(),
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

/// `t.into()`: the similarity of translation `t`, identity rotation and scaling 1. Upstream:
/// `SubsetOf<Similarity2> for Translation2` (`nalgebra::convert(t)`).
pub impl Similarity2FromTranslation2<
    T, impl R: Real<T>, +Drop<T>,
> of Into<Translation2<T>, Similarity2<T>> {
    #[inline(always)]
    fn into(self: Translation2<T>) -> Similarity2<T> {
        Similarity2 {
            isometry: Isometry2 {
                rotation: UnitComplex { re: R::one(), im: R::zero() }, translation: self,
            },
            scaling: R::one(),
        }
    }
}
