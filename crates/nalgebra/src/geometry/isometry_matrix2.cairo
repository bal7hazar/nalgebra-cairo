//! `IsometryMatrix2`: a 2D rigid-body transform whose rotation is a `Rotation2` MATRIX (upstream
//! `nalgebra::IsometryMatrix2`, which is `Isometry<T, Rotation2<T>, 2>`).
//!
//! Upstream has one generic `Isometry<T, R, D>`; Cairo has no generic-rotation struct with the
//! fused kernels of each rotation type, so the rotation-matrix instance is a struct of its own with
//! the same method set as `Isometry2` (whose rotation is a `UnitComplex`), the same field names
//! (`rotation`, `translation`) and the same semantics. The two convert into each other with
//! `.into()` (`nalgebra::convert` upstream, exact both ways: the complex IS the first column).
//!
//! - `IsometryMatrix2Trait` / `IsometryMatrix2Impl`: everything algebraic (composition, inverse,
//!   transforms, the in-place `append_*_mut`, the heterogeneous operators as `mul_<rhs>` /
//!   `div_<rhs>` methods, homogeneous form, comparisons, `cast`), over a `Real` scalar;
//! - `IsometryMatrix2AngleTrait` / `IsometryMatrix2AngleImpl`: the constructors through an angle
//! and
//!   `lerp_slerp`, which additionally need `simba::scalar::Transcendental`;
//! - the operator / conversion impls (`*`, `/`, `*=`, `/=`, `Default`, `One`, `From`).
//!
//! **Prefer `Isometry2` in hot code**: composing two rotation matrices costs four fused kernels
//! where the unit complex form costs two (`rotation2` module documentation); the transforms cost
//! exactly the same. Every "rotate then translate" goes through the fused `rotate_translate`
//! kernel (one rounding per component), bit for bit what rotating then adding gives, since
//! `floor(x + t) = floor(x) + t` for an integral `t` in raw units.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::{Real, Transcendental};
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::unit::Unit;
use crate::base::vector2::Vector2;
use super::isometry2::Isometry2;
use super::rotation2::{Rotation2, Rotation2AngleTrait, Rotation2Trait};
use super::similarity_matrix2::{SimilarityMatrix2, SimilarityMatrix2Trait};
use super::translation2::{Translation2, Translation2Trait};
use super::unit_complex::UnitComplex;

/// A 2D direct isometry whose rotation is stored as a matrix: the rotation `rotation` followed by
/// the translation `translation` (upstream's field names and order).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct IsometryMatrix2<T> {
    pub rotation: Rotation2<T>,
    pub translation: Translation2<T>,
}

/// Operations of `IsometryMatrix2<T>` that need no trigonometry, over a `Real` scalar. By value,
/// unrolled, no loop.
#[generate_trait]
pub impl IsometryMatrix2Impl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
> of IsometryMatrix2Trait<T> {
    // --- construction and parts ---------------------------------------------------------------

    /// The identity isometry. Exact. Upstream: `Isometry::identity`.
    #[inline(always)]
    fn identity() -> IsometryMatrix2<T> {
        IsometryMatrix2 {
            rotation: Rotation2Trait::identity(),
            translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
        }
    }

    /// The isometry `translation ∘ rotation` (rotate first, then translate). Exact. Upstream:
    /// `Isometry::from_parts`.
    #[inline(always)]
    fn from_parts(translation: Translation2<T>, rotation: Rotation2<T>) -> IsometryMatrix2<T> {
        IsometryMatrix2 { rotation, translation }
    }

    /// The pure translation by `(x, y)`. Exact (upstream builds `Rotation2::new(0)`, which is the
    /// identity exactly). Upstream: `IsometryMatrix2::translation`.
    #[inline(always)]
    fn translation(x: T, y: T) -> IsometryMatrix2<T> {
        IsometryMatrix2 {
            rotation: Rotation2Trait::identity(),
            translation: Translation2 { vector: Vector2 { x, y } },
        }
    }

    /// The rotation `r` about the point `p` (which stays fixed): translation `r · (-p) + p`
    /// through one `rotate_translate` (upstream: `r.transform_vector(-p) + p`, the same bits).
    /// Upstream: `Isometry::rotation_wrt_point`.
    #[inline(always)]
    fn rotation_wrt_point(r: Rotation2<T>, p: Point2<T>) -> IsometryMatrix2<T> {
        IsometryMatrix2 {
            rotation: r,
            translation: Translation2 {
                vector: IsometryMatrix2InternalTrait::rotate_translate(
                    r, Vector2 { x: -p.x, y: -p.y }, Vector2 { x: p.x, y: p.y },
                ),
            },
        }
    }

    // --- inverse and composition --------------------------------------------------------------

    /// The inverse isometry: the transposed rotation (exact) and the translation
    /// `rotationᵀ · (-translation)` (two fused kernels on the negated vector, upstream's order).
    /// Panics on overflow (`-MIN`). Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: IsometryMatrix2<T>) -> IsometryMatrix2<T> {
        let v = Vector2 { x: -self.translation.vector.x, y: -self.translation.vector.y };
        IsometryMatrix2 {
            rotation: self.rotation.inverse(),
            translation: Translation2 { vector: self.rotation.inverse_transform_vector(v) },
        }
    }

    /// `self⁻¹ * other` without materialising the inverse: translation `rotationᵀ ·
    /// (other.translation - self.translation)`, rotation `rotationᵀ · other.rotation` (the
    /// transpose is exact, then four fused kernels). Upstream: `inv_mul`.
    fn inv_mul(self: IsometryMatrix2<T>, other: IsometryMatrix2<T>) -> IsometryMatrix2<T> {
        let d = Vector2 {
            x: other.translation.vector.x - self.translation.vector.x,
            y: other.translation.vector.y - self.translation.vector.y,
        };
        IsometryMatrix2 {
            rotation: self.rotation.inverse() * other.rotation,
            translation: Translation2 { vector: self.rotation.inverse_transform_vector(d) },
        }
    }

    // --- transforms ----------------------------------------------------------------------------

    /// `self * p = rotation · p + translation`, one `rotate_translate`. Upstream:
    /// `transform_point` (`iso * p`).
    #[inline(always)]
    fn transform_point(self: IsometryMatrix2<T>, p: Point2<T>) -> Point2<T> {
        let c = IsometryMatrix2InternalTrait::rotate_translate(
            self.rotation, Vector2 { x: p.x, y: p.y }, self.translation.vector,
        );
        Point2 { x: c.x, y: c.y }
    }

    /// `rotation · v` (the translation cancels on a displacement). Upstream: `transform_vector`
    /// (`iso * v`).
    #[inline(always)]
    fn transform_vector(self: IsometryMatrix2<T>, v: Vector2<T>) -> Vector2<T> {
        self.rotation.transform_vector(v)
    }

    /// `rotation · v` for a unit vector, re-wrapped WITHOUT renormalising (upstream's
    /// `Unit::new_unchecked`). Upstream: `Mul<Unit<Vector2>> for Isometry` (`iso * v`).
    #[inline(always)]
    fn transform_unit_vector(self: IsometryMatrix2<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>> {
        Unit { value: self.rotation.transform_vector(v.value) }
    }

    /// `self⁻¹ * p = rotationᵀ · (p - translation)`: one exact subtraction, two fused
    /// kernels.
    /// Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: IsometryMatrix2<T>, p: Point2<T>) -> Point2<T> {
        let d = Point2 { x: p.x - self.translation.vector.x, y: p.y - self.translation.vector.y };
        self.rotation.inverse_transform_point(d)
    }

    /// `rotationᵀ · v`. Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: IsometryMatrix2<T>, v: Vector2<T>) -> Vector2<T> {
        self.rotation.inverse_transform_vector(v)
    }

    /// `rotationᵀ · v` for a unit vector, not renormalised. Upstream:
    /// `inverse_transform_unit_vector`.
    #[inline(always)]
    fn inverse_transform_unit_vector(
        self: IsometryMatrix2<T>, v: Unit<Vector2<T>>,
    ) -> Unit<Vector2<T>> {
        Unit { value: self.rotation.inverse_transform_vector(v.value) }
    }

    // --- append (in place) and the operator forms ----------------------------------------------

    /// `Translation(t) ∘ self`: the translation shifted by `t` (exact), in place. Upstream:
    /// `append_translation_mut`.
    #[inline(always)]
    fn append_translation_mut(ref self: IsometryMatrix2<T>, t: Translation2<T>) {
        self =
            IsometryMatrix2 {
                rotation: self.rotation,
                translation: Translation2 {
                    vector: Vector2 {
                        x: self.translation.vector.x + t.vector.x,
                        y: self.translation.vector.y + t.vector.y,
                    },
                },
            };
    }

    /// `self * t`: the translation shifted by `rotation · t` (one `rotate_translate`). Upstream:
    /// `Mul<Translation> for Isometry` (a named method: Cairo's `Mul` is homogeneous).
    #[inline(always)]
    fn mul_translation(self: IsometryMatrix2<T>, t: Translation2<T>) -> IsometryMatrix2<T> {
        IsometryMatrix2 {
            rotation: self.rotation,
            translation: Translation2 {
                vector: IsometryMatrix2InternalTrait::rotate_translate(
                    self.rotation, t.vector, self.translation.vector,
                ),
            },
        }
    }

    /// `Rotation(r) ∘ self`: rotation `r · rotation` (four fused kernels) and translation
    /// `r · translation`, in place. Upstream: `append_rotation_mut`.
    #[inline(always)]
    fn append_rotation_mut(ref self: IsometryMatrix2<T>, r: Rotation2<T>) {
        self =
            IsometryMatrix2 {
                rotation: r * self.rotation,
                translation: Translation2 { vector: r.transform_vector(self.translation.vector) },
            };
    }

    /// The rotation `r` applied about the point `p`, in place: translation
    /// `r · (translation - p) + p` (one exact subtraction and one `rotate_translate`), rotation
    /// `r · rotation`. Upstream: `append_rotation_wrt_point_mut`.
    fn append_rotation_wrt_point_mut(ref self: IsometryMatrix2<T>, r: Rotation2<T>, p: Point2<T>) {
        let d = Vector2 { x: self.translation.vector.x - p.x, y: self.translation.vector.y - p.y };
        self =
            IsometryMatrix2 {
                rotation: r * self.rotation,
                translation: Translation2 {
                    vector: IsometryMatrix2InternalTrait::rotate_translate(
                        r, d, Vector2 { x: p.x, y: p.y },
                    ),
                },
            };
    }

    /// The rotation `r` applied about the isometry's own centre: only the rotation changes
    /// (`r · rotation`), in place. Upstream: `append_rotation_wrt_center_mut`.
    #[inline(always)]
    fn append_rotation_wrt_center_mut(ref self: IsometryMatrix2<T>, r: Rotation2<T>) {
        self = IsometryMatrix2 { rotation: r * self.rotation, translation: self.translation };
    }

    /// `self * r`: a rotation applied BEFORE `self` (the translation is unchanged, the rotation
    /// becomes `rotation · r`). Upstream: `Mul<Rotation> for Isometry<T, Rotation<T, D>, D>` (a
    /// named method: Cairo's `Mul` is homogeneous).
    #[inline(always)]
    fn mul_rotation(self: IsometryMatrix2<T>, r: Rotation2<T>) -> IsometryMatrix2<T> {
        IsometryMatrix2 { rotation: self.rotation * r, translation: self.translation }
    }

    /// `self / r = self * r⁻¹`: the rotation becomes `rotation / r` (the product with the
    /// transpose). Upstream: `Div<Rotation> for Isometry<T, Rotation<T, D>, D>`.
    #[inline(always)]
    fn div_rotation(self: IsometryMatrix2<T>, r: Rotation2<T>) -> IsometryMatrix2<T> {
        IsometryMatrix2 { rotation: self.rotation / r, translation: self.translation }
    }

    /// `self * sim`: the similarity `(self * sim.isometry, sim.scaling)`. Upstream:
    /// `Mul<Similarity> for Isometry`.
    #[inline(always)]
    fn mul_similarity(self: IsometryMatrix2<T>, sim: SimilarityMatrix2<T>) -> SimilarityMatrix2<T> {
        SimilarityMatrix2 { isometry: self * sim.isometry, scaling: sim.scaling }
    }

    /// `self / sim = self * sim⁻¹` (upstream's formula: the inverse similarity is materialised).
    /// Panics like `SimilarityMatrix2::inverse`. Upstream: `Div<Similarity> for Isometry`.
    #[inline(always)]
    fn div_similarity(self: IsometryMatrix2<T>, sim: SimilarityMatrix2<T>) -> SimilarityMatrix2<T> {
        Self::mul_similarity(self, sim.inverse())
    }

    // --- conversions and comparisons -----------------------------------------------------------

    /// The 3x3 homogeneous matrix `[[m11, m12, tx], [m21, m22, ty], [0, 0, 1]]`. Exact (copies).
    /// Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: IsometryMatrix2<T>) -> Matrix3<T> {
        let m = self.rotation.matrix;
        Matrix3 {
            m11: m.m11,
            m21: m.m21,
            m31: R::zero(),
            m12: m.m12,
            m22: m.m22,
            m32: R::zero(),
            m13: self.translation.vector.x,
            m23: self.translation.vector.y,
            m33: R::one(),
        }
    }

    /// Alias of `to_homogeneous`. Upstream: `Isometry::to_matrix`.
    #[inline(always)]
    fn to_matrix(self: IsometryMatrix2<T>) -> Matrix3<T> {
        Self::to_homogeneous(self)
    }

    /// `true` when the translations and the rotation matrices are within `ulps` smallest units
    /// of each other, component by component. Upstream: `approx::AbsDiffEq::abs_diff_eq` (DESIGN
    /// D3).
    #[inline(always)]
    fn abs_diff_eq(self: IsometryMatrix2<T>, other: IsometryMatrix2<T>, ulps: u64) -> bool {
        self.translation.abs_diff_eq(other.translation, ulps)
            && self.rotation.abs_diff_eq(other.rotation, ulps)
    }

    /// `true` when the translations and the rotations are `relative_eq` (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: IsometryMatrix2<T>, other: IsometryMatrix2<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        self.translation.relative_eq(other.translation, epsilon, max_relative)
            && self.rotation.relative_eq(other.rotation, epsilon, max_relative)
    }

    /// `true` when the translations and the rotations are `ulps_eq` (see
    /// `QuaternionTrait::ulps_eq`). Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(
        self: IsometryMatrix2<T>, other: IsometryMatrix2<T>, epsilon: u64, max_ulps: u32,
    ) -> bool {
        self.translation.ulps_eq(other.translation, epsilon, max_ulps)
            && self.rotation.ulps_eq(other.rotation, epsilon, max_ulps)
    }

    /// The same isometry with every scalar converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `IsometryMatrix2::cast` (and `SubsetOf<Isometry>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: IsometryMatrix2<T>) -> IsometryMatrix2<U> {
        IsometryMatrix2 { rotation: self.rotation.cast(), translation: self.translation.cast() }
    }
}

/// Crate-internal kernel of `IsometryMatrix2<T>`: the fused "rotate then translate".
#[generate_trait]
pub(crate) impl IsometryMatrix2InternalImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of IsometryMatrix2InternalTrait<T> {
    /// `r · v + t`: the two products AND the translation accumulated exactly, floored once per
    /// component. The same bits as `r.transform_vector(v) + t` (`floor(x + t) = floor(x) + t`
    /// for an integral `t`), without the overflow check of the `Fixed` addition; it can only
    /// overflow on the result. Not an upstream method (upstream writes `rotation * v +
    /// translation`).
    #[inline(always)]
    fn rotate_translate(r: Rotation2<T>, v: Vector2<T>, t: Vector2<T>) -> Vector2<T> {
        let m = r.matrix;
        Vector2 {
            x: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m11, v.x), m.m12, v.y), t.x,
                ),
            ),
            y: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), m.m21, v.x), m.m22, v.y), t.y,
                ),
            ),
        }
    }
}

/// Operations of `IsometryMatrix2<T>` that go through an angle (one `sin_cos` or more), hence
/// their own trait.
#[generate_trait]
pub impl IsometryMatrix2AngleImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
> of IsometryMatrix2AngleTrait<T> {
    /// The isometry that rotates by `angle` (radians, `Rotation2::new`: one `sin_cos`) then
    /// translates by `translation`. Upstream: `IsometryMatrix2::new(translation, angle)`.
    #[inline(always)]
    fn new(translation: Vector2<T>, angle: T) -> IsometryMatrix2<T> {
        IsometryMatrix2 {
            rotation: Rotation2AngleTrait::new(angle),
            translation: Translation2 { vector: translation },
        }
    }

    /// The pure rotation of `angle` radians about the origin. Upstream:
    /// `IsometryMatrix2::rotation`.
    #[inline(always)]
    fn rotation(angle: T) -> IsometryMatrix2<T> {
        IsometryMatrix2 {
            rotation: Rotation2AngleTrait::new(angle),
            translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
        }
    }

    /// Interpolation between two poses: the translations linearly (one fused `lerp` per
    /// component), the rotations by `Rotation2::slerp` (shortest arc, constant angular
    /// velocity). `t` is not clamped. Upstream: `IsometryMatrix2::lerp_slerp`.
    fn lerp_slerp(self: IsometryMatrix2<T>, other: IsometryMatrix2<T>, t: T) -> IsometryMatrix2<T> {
        IsometryMatrix2 {
            rotation: self.rotation.slerp(other.rotation, t),
            translation: Translation2 {
                vector: Vector2 {
                    x: R::lerp(self.translation.vector.x, other.translation.vector.x, t),
                    y: R::lerp(self.translation.vector.y, other.translation.vector.y, t),
                },
            },
        }
    }
}

/// `a * b`: the composition, `b` applied first: rotation `a.rotation · b.rotation` (four fused
/// kernels) and translation `a.translation + a.rotation · b.translation` (one
/// `rotate_translate`). Upstream: `Mul<Isometry> for Isometry`.
pub impl IsometryMatrix2Mul<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of Mul<IsometryMatrix2<T>> {
    fn mul(lhs: IsometryMatrix2<T>, rhs: IsometryMatrix2<T>) -> IsometryMatrix2<T> {
        IsometryMatrix2 {
            rotation: lhs.rotation * rhs.rotation,
            translation: Translation2 {
                vector: IsometryMatrix2InternalTrait::rotate_translate(
                    lhs.rotation, rhs.translation.vector, lhs.translation.vector,
                ),
            },
        }
    }
}

/// `a / b = a * b⁻¹`, upstream's formula (the inverse is materialised, then composed). Upstream:
/// `Div<Isometry> for Isometry`.
pub impl IsometryMatrix2Div<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
> of Div<IsometryMatrix2<T>> {
    #[inline(always)]
    fn div(lhs: IsometryMatrix2<T>, rhs: IsometryMatrix2<T>) -> IsometryMatrix2<T> {
        lhs * rhs.inverse()
    }
}

/// `iso *= t`: `iso = iso * t` (`mul_translation`). Upstream: `MulAssign<Translation> for
/// Isometry`.
pub impl IsometryMatrix2MulAssignTranslation2<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
> of MulAssign<IsometryMatrix2<T>, Translation2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: IsometryMatrix2<T>, rhs: Translation2<T>) {
        self = self.mul_translation(rhs);
    }
}

/// `a *= b`: `a = a * b`. Upstream: `MulAssign<Isometry> for Isometry`.
pub impl IsometryMatrix2MulAssign<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of MulAssign<IsometryMatrix2<T>, IsometryMatrix2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: IsometryMatrix2<T>, rhs: IsometryMatrix2<T>) {
        self = self * rhs;
    }
}

/// `a /= b`: `a = a * b⁻¹`. Upstream: `DivAssign<Isometry> for Isometry`.
pub impl IsometryMatrix2DivAssign<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
> of DivAssign<IsometryMatrix2<T>, IsometryMatrix2<T>> {
    #[inline(always)]
    fn div_assign(ref self: IsometryMatrix2<T>, rhs: IsometryMatrix2<T>) {
        self = self * rhs.inverse();
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Isometry`.
pub impl IsometryMatrix2Default<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Default<IsometryMatrix2<T>> {
    #[inline(always)]
    fn default() -> IsometryMatrix2<T> {
        IsometryMatrix2 {
            rotation: Rotation2 {
                matrix: Matrix2 { m11: R::one(), m21: R::zero(), m12: R::zero(), m22: R::one() },
            },
            translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
        }
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Isometry`.
pub impl IsometryMatrix2One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<IsometryMatrix2<T>> {
    #[inline(always)]
    fn one() -> IsometryMatrix2<T> {
        IsometryMatrix2Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @IsometryMatrix2<T>) -> bool {
        *self == IsometryMatrix2Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @IsometryMatrix2<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `t.into()`: the pure translation. Upstream: `From<Translation> for Isometry`.
pub impl IsometryMatrix2FromTranslation2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Translation2<T>, IsometryMatrix2<T>> {
    #[inline(always)]
    fn into(self: Translation2<T>) -> IsometryMatrix2<T> {
        let mut iso = IsometryMatrix2Default::<T>::default();
        iso.translation = self;
        iso
    }
}

/// `v.into()`: the pure translation by the vector `v`. Upstream: `From<SVector<T, 2>> for
/// Isometry`.
pub impl IsometryMatrix2FromVector2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Vector2<T>, IsometryMatrix2<T>> {
    #[inline(always)]
    fn into(self: Vector2<T>) -> IsometryMatrix2<T> {
        let mut iso = IsometryMatrix2Default::<T>::default();
        iso.translation = Translation2 { vector: self };
        iso
    }
}

/// `p.into()`: the pure translation by the coordinates of `p`. Upstream: `From<Point<T, 2>> for
/// Isometry`.
pub impl IsometryMatrix2FromPoint2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Point2<T>, IsometryMatrix2<T>> {
    #[inline(always)]
    fn into(self: Point2<T>) -> IsometryMatrix2<T> {
        let mut iso = IsometryMatrix2Default::<T>::default();
        iso.translation = Translation2 { vector: Vector2 { x: self.x, y: self.y } };
        iso
    }
}

/// `[x, y].into()`: the pure translation by `(x, y)`. Upstream: `From<[T; 2]> for Isometry`.
pub impl IsometryMatrix2FromArray<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<[T; 2], IsometryMatrix2<T>> {
    #[inline(always)]
    fn into(self: [T; 2]) -> IsometryMatrix2<T> {
        let [x, y] = self;
        let mut iso = IsometryMatrix2Default::<T>::default();
        iso.translation = Translation2 { vector: Vector2 { x, y } };
        iso
    }
}

/// `iso.into()`: the same isometry with its rotation as a unit complex (the first column of the
/// matrix, exact). Upstream: `SubsetOf<Isometry2> for IsometryMatrix2` (`nalgebra::convert`).
pub impl Isometry2FromIsometryMatrix2<
    T, +Copy<T>, +Drop<T>,
> of Into<IsometryMatrix2<T>, Isometry2<T>> {
    #[inline(always)]
    fn into(self: IsometryMatrix2<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: UnitComplex { re: self.rotation.matrix.m11, im: self.rotation.matrix.m21 },
            translation: self.translation,
        }
    }
}

/// `iso.into()`: the same isometry with its rotation as a matrix (`[[re, -im], [im, re]]`,
/// exact). Upstream: `SubsetOf<IsometryMatrix2> for Isometry2` (`nalgebra::convert`).
pub impl IsometryMatrix2FromIsometry2<
    T, +Copy<T>, +Drop<T>, +Neg<T>,
> of Into<Isometry2<T>, IsometryMatrix2<T>> {
    #[inline(always)]
    fn into(self: Isometry2<T>) -> IsometryMatrix2<T> {
        let (re, im) = (self.rotation.re, self.rotation.im);
        IsometryMatrix2 {
            rotation: Rotation2 { matrix: Matrix2 { m11: re, m21: im, m12: -im, m22: re } },
            translation: self.translation,
        }
    }
}

/// `iso.into()`: the similarity of scaling 1. Upstream: `SubsetOf<Similarity> for Isometry`
/// (`nalgebra::convert`).
pub impl SimilarityMatrix2FromIsometryMatrix2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<IsometryMatrix2<T>, SimilarityMatrix2<T>> {
    #[inline(always)]
    fn into(self: IsometryMatrix2<T>) -> SimilarityMatrix2<T> {
        SimilarityMatrix2 { isometry: self, scaling: R::one() }
    }
}
