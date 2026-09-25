//! `SimilarityMatrix2`: a 2D uniform scaling, followed by a rotation stored as a `Rotation2`
//! MATRIX, followed by a translation (upstream `nalgebra::SimilarityMatrix2`, which is
//! `Similarity<T, Rotation2<T>, 2>`).
//!
//! `sim * p = translation + scaling * (rotation * p)`. The rotation-matrix instance of upstream's
//! generic `Similarity` is a struct of its own, with the method set, field names and semantics of
//! `Similarity2` (whose rotation is a `UnitComplex`); the two convert into each other exactly with
//! `.into()`. The scaling factor must be nonzero: constructors and scaling mutators panic with
//! `nalgebra: zero scale` on zero (`similarity2::errors::ZERO_SCALING`).
//!
//! Numeric contract: the rotations go through the `Rotation2` fused kernels, the final
//! scale-plus-translation is one wide accumulation per component (`scale_translate`, shared with
//! `Similarity2`), preserving upstream's order (rotate, then scale, then translate).

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::{Real, Transcendental};
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;
use super::isometry_matrix2::{IsometryMatrix2, IsometryMatrix2Trait};
use super::quaternion::ApproxEqTrait;
use super::rotation2::{Rotation2, Rotation2AngleTrait, Rotation2Trait};
use super::similarity2::errors::ZERO_SCALING;
use super::similarity2::{Similarity2, Similarity2InternalTrait};
use super::translation2::Translation2;

/// A 2D direct similarity whose rotation is a matrix: the uniform scale `scaling`, then
/// `isometry.rotation`, then `isometry.translation` (upstream's fields).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct SimilarityMatrix2<T> {
    pub isometry: IsometryMatrix2<T>,
    pub scaling: T,
}

/// Operations of `SimilarityMatrix2<T>` that need no trigonometry, over a `Real` scalar.
#[generate_trait]
pub impl SimilarityMatrix2Impl<
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
> of SimilarityMatrix2Trait<T> {
    /// The identity similarity (unit scale, no rotation, no translation). Exact. Upstream:
    /// `Similarity::identity`.
    #[inline(always)]
    fn identity() -> SimilarityMatrix2<T> {
        SimilarityMatrix2 { isometry: IsometryMatrix2Trait::identity(), scaling: R::one() }
    }

    /// The similarity from `translation`, `rotation` and nonzero `scaling`. Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `Similarity::from_parts`.
    #[inline(always)]
    fn from_parts(
        translation: Translation2<T>, rotation: Rotation2<T>, scaling: T,
    ) -> SimilarityMatrix2<T> {
        Self::from_isometry(IsometryMatrix2Trait::from_parts(translation, rotation), scaling)
    }

    /// The similarity from an isometry and a nonzero scale. Panics with `nalgebra: zero scale` on
    /// zero. Upstream: `Similarity::from_isometry`.
    #[inline(always)]
    fn from_isometry(isometry: IsometryMatrix2<T>, scaling: T) -> SimilarityMatrix2<T> {
        if scaling == R::zero() {
            core::panic_with_felt252(ZERO_SCALING);
        }
        SimilarityMatrix2 { isometry, scaling }
    }

    /// The pure nonzero uniform scale. Panics with `nalgebra: zero scale` on zero. Upstream:
    /// `Similarity::from_scaling`.
    #[inline(always)]
    fn from_scaling(scaling: T) -> SimilarityMatrix2<T> {
        Self::from_isometry(IsometryMatrix2Trait::identity(), scaling)
    }

    /// The rotation `r` about the point `p` followed by nothing else, with the scale `scaling`:
    /// translation `r · (-p) + p` (upstream's formula, one `rotate_translate`). Panics with
    /// `nalgebra: zero scale` on a zero scale. Upstream: `Similarity::rotation_wrt_point`.
    #[inline(always)]
    fn rotation_wrt_point(r: Rotation2<T>, p: Point2<T>, scaling: T) -> SimilarityMatrix2<T> {
        Self::from_isometry(IsometryMatrix2Trait::rotation_wrt_point(r, p), scaling)
    }

    /// The stored scaling factor. Exact. Upstream: `scaling`.
    #[inline(always)]
    fn scaling(self: SimilarityMatrix2<T>) -> T {
        self.scaling
    }

    /// Sets a new nonzero scale, in place. Panics with `nalgebra: zero scale` on zero. Upstream:
    /// `set_scaling`.
    #[inline(always)]
    fn set_scaling(ref self: SimilarityMatrix2<T>, scaling: T) {
        self = Self::from_isometry(self.isometry, scaling);
    }

    /// Applies the scale `s` BEFORE `self`: only the stored scale changes (one product). Panics
    /// on a zero `s`. Upstream: `prepend_scaling`.
    #[inline(always)]
    fn prepend_scaling(self: SimilarityMatrix2<T>, s: T) -> SimilarityMatrix2<T> {
        if s == R::zero() {
            core::panic_with_felt252(ZERO_SCALING);
        }
        SimilarityMatrix2 { isometry: self.isometry, scaling: self.scaling * s }
    }

    /// Applies the scale `s` AFTER `self`: the translation and the scale are multiplied by `s`.
    /// Panics on a zero `s`. Upstream: `append_scaling`.
    #[inline(always)]
    fn append_scaling(self: SimilarityMatrix2<T>, s: T) -> SimilarityMatrix2<T> {
        if s == R::zero() {
            core::panic_with_felt252(ZERO_SCALING);
        }
        let t = self.isometry.translation.vector;
        SimilarityMatrix2 {
            isometry: IsometryMatrix2 {
                rotation: self.isometry.rotation,
                translation: Translation2 { vector: Vector2 { x: t.x * s, y: t.y * s } },
            },
            scaling: self.scaling * s,
        }
    }

    /// The inverse similarity, like `Similarity2::inverse`: the inverse isometry, its translation
    /// divided by the scale (one correctly rounded quotient per component), the scale inverted.
    /// Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: SimilarityMatrix2<T>) -> SimilarityMatrix2<T> {
        let inv_iso = self.isometry.inverse();
        let t = inv_iso.translation.vector;
        SimilarityMatrix2 {
            isometry: IsometryMatrix2 {
                rotation: inv_iso.rotation,
                translation: Translation2 {
                    vector: Vector2 { x: R::div(t.x, self.scaling), y: R::div(t.y, self.scaling) },
                },
            },
            scaling: R::div(R::one(), self.scaling),
        }
    }

    /// `self = self.inverse()` in place: the by-value form is the cheapest (a rotation and a
    /// translation to rebuild, nothing to reuse), so the bits are those of `inverse`. Panics as
    /// `inverse` does. Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: SimilarityMatrix2<T>) {
        self = Self::inverse(self);
    }

    /// `self = self.prepend_scaling(s)` in place: only the stored scale is multiplied by `s`.
    /// Panics with `nalgebra: zero scale` on a zero `s`, and on overflow. Upstream:
    /// `prepend_scaling_mut`.
    #[inline(always)]
    fn prepend_scaling_mut(ref self: SimilarityMatrix2<T>, s: T) {
        self = Self::prepend_scaling(self, s);
    }

    /// `self = self.append_scaling(s)` in place: the translation and the scale are multiplied by
    /// `s`. Panics with `nalgebra: zero scale` on a zero `s`, and on overflow. Upstream:
    /// `append_scaling_mut`.
    #[inline(always)]
    fn append_scaling_mut(ref self: SimilarityMatrix2<T>, s: T) {
        self = Self::append_scaling(self, s);
    }

    /// `self * p = translation + scaling * (rotation · p)`. Upstream: `transform_point`.
    #[inline(always)]
    fn transform_point(self: SimilarityMatrix2<T>, p: Point2<T>) -> Point2<T> {
        let c = Similarity2InternalTrait::scale_translate(
            self.isometry.rotation.transform_vector(Vector2 { x: p.x, y: p.y }),
            self.scaling,
            self.isometry.translation.vector,
        );
        Point2 { x: c.x, y: c.y }
    }

    /// `scaling * (rotation · v)`. Upstream: `transform_vector`.
    #[inline(always)]
    fn transform_vector(self: SimilarityMatrix2<T>, v: Vector2<T>) -> Vector2<T> {
        let r = self.isometry.rotation.transform_vector(v);
        Vector2 { x: r.x * self.scaling, y: r.y * self.scaling }
    }

    /// `self⁻¹ * p = rotationᵀ · (p - translation) / scaling`. Upstream:
    /// `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: SimilarityMatrix2<T>, p: Point2<T>) -> Point2<T> {
        let c = self.isometry.inverse_transform_point(p);
        Point2 { x: R::div(c.x, self.scaling), y: R::div(c.y, self.scaling) }
    }

    /// `rotationᵀ · v / scaling`. Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: SimilarityMatrix2<T>, v: Vector2<T>) -> Vector2<T> {
        let c = self.isometry.inverse_transform_vector(v);
        Vector2 { x: R::div(c.x, self.scaling), y: R::div(c.y, self.scaling) }
    }

    /// `Translation(t) ∘ self`, in place. Upstream: `append_translation_mut`.
    #[inline(always)]
    fn append_translation_mut(ref self: SimilarityMatrix2<T>, t: Translation2<T>) {
        let mut isometry = self.isometry;
        isometry.append_translation_mut(t);
        self = SimilarityMatrix2 { isometry, scaling: self.scaling };
    }

    /// `self * t`: the translation shifted by `scaling * (rotation · t)`. Upstream:
    /// `Mul<Translation> for Similarity` (a named method: Cairo's `Mul` is homogeneous).
    #[inline(always)]
    fn mul_translation(self: SimilarityMatrix2<T>, t: Translation2<T>) -> SimilarityMatrix2<T> {
        SimilarityMatrix2 {
            isometry: IsometryMatrix2 {
                rotation: self.isometry.rotation,
                translation: Translation2 {
                    vector: Similarity2InternalTrait::scale_translate(
                        self.isometry.rotation.transform_vector(t.vector),
                        self.scaling,
                        self.isometry.translation.vector,
                    ),
                },
            },
            scaling: self.scaling,
        }
    }

    /// `Rotation(r) ∘ self`, in place (the scale is unchanged). Upstream: `append_rotation_mut`.
    #[inline(always)]
    fn append_rotation_mut(ref self: SimilarityMatrix2<T>, r: Rotation2<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_mut(r);
        self = SimilarityMatrix2 { isometry, scaling: self.scaling };
    }

    /// The rotation `r` appended about the point `p`, in place. Upstream:
    /// `append_rotation_wrt_point_mut`.
    #[inline(always)]
    fn append_rotation_wrt_point_mut(
        ref self: SimilarityMatrix2<T>, r: Rotation2<T>, p: Point2<T>,
    ) {
        let mut isometry = self.isometry;
        isometry.append_rotation_wrt_point_mut(r, p);
        self = SimilarityMatrix2 { isometry, scaling: self.scaling };
    }

    /// The rotation `r` appended about the similarity's centre, in place. Upstream:
    /// `append_rotation_wrt_center_mut`.
    #[inline(always)]
    fn append_rotation_wrt_center_mut(ref self: SimilarityMatrix2<T>, r: Rotation2<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_wrt_center_mut(r);
        self = SimilarityMatrix2 { isometry, scaling: self.scaling };
    }

    /// `self * r`: rotation before the similarity. Upstream: `Mul<Rotation> for
    /// Similarity<T, Rotation<T, D>, D>`.
    #[inline(always)]
    fn mul_rotation(self: SimilarityMatrix2<T>, r: Rotation2<T>) -> SimilarityMatrix2<T> {
        SimilarityMatrix2 { isometry: self.isometry.mul_rotation(r), scaling: self.scaling }
    }

    /// `self / r = self * r⁻¹`. Upstream: `Div<Rotation> for Similarity<T, Rotation<T, D>, D>`.
    #[inline(always)]
    fn div_rotation(self: SimilarityMatrix2<T>, r: Rotation2<T>) -> SimilarityMatrix2<T> {
        SimilarityMatrix2 { isometry: self.isometry.div_rotation(r), scaling: self.scaling }
    }

    /// `self * iso`: translation `translation + scaling * (rotation · iso.translation)`, rotation
    /// `rotation · iso.rotation`, the same scale. Upstream: `Mul<Isometry> for Similarity`.
    #[inline(always)]
    fn mul_isometry(self: SimilarityMatrix2<T>, iso: IsometryMatrix2<T>) -> SimilarityMatrix2<T> {
        SimilarityMatrix2 {
            isometry: IsometryMatrix2 {
                rotation: self.isometry.rotation * iso.rotation,
                translation: Translation2 {
                    vector: Similarity2InternalTrait::scale_translate(
                        self.isometry.rotation.transform_vector(iso.translation.vector),
                        self.scaling,
                        self.isometry.translation.vector,
                    ),
                },
            },
            scaling: self.scaling,
        }
    }

    /// `self / iso = self * iso⁻¹` (upstream's formula). Upstream: `Div<Isometry> for
    /// Similarity`.
    #[inline(always)]
    fn div_isometry(self: SimilarityMatrix2<T>, iso: IsometryMatrix2<T>) -> SimilarityMatrix2<T> {
        Self::mul_isometry(self, iso.inverse())
    }

    /// The homogeneous matrix: the rotation block times the scale (one product per entry), the
    /// translation, then `(0, 0, 1)`. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: SimilarityMatrix2<T>) -> Matrix3<T> {
        let (m, s) = (self.isometry.rotation.matrix, self.scaling);
        Matrix3 {
            m11: m.m11 * s,
            m21: m.m21 * s,
            m31: R::zero(),
            m12: m.m12 * s,
            m22: m.m22 * s,
            m32: R::zero(),
            m13: self.isometry.translation.vector.x,
            m23: self.isometry.translation.vector.y,
            m33: R::one(),
        }
    }

    /// Component-wise `abs_diff_eq` of the isometry and the scale, in ulp. Upstream:
    /// `AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: SimilarityMatrix2<T>, other: SimilarityMatrix2<T>, ulps: u64) -> bool {
        self.isometry.abs_diff_eq(other.isometry, ulps)
            && R::abs_diff_eq(self.scaling, other.scaling, ulps)
    }

    /// `relative_eq` of the isometries and of the scales. Upstream:
    /// `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: SimilarityMatrix2<T>, other: SimilarityMatrix2<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        self.isometry.relative_eq(other.isometry, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.scaling, other.scaling, epsilon, max_relative)
    }

    /// `ulps_eq` of the isometries and of the scales. Upstream: `approx::UlpsEq::ulps_eq`
    /// (DESIGN D3).
    fn ulps_eq(
        self: SimilarityMatrix2<T>, other: SimilarityMatrix2<T>, epsilon: u64, max_ulps: u32,
    ) -> bool {
        self.isometry.ulps_eq(other.isometry, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.scaling, other.scaling, epsilon, max_ulps)
    }

    /// The same similarity with every scalar converted by `Into<T, U>`. Upstream:
    /// `SimilarityMatrix2::cast` (and `SubsetOf<Similarity>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: SimilarityMatrix2<T>) -> SimilarityMatrix2<U> {
        SimilarityMatrix2 { isometry: self.isometry.cast(), scaling: self.scaling.into() }
    }
}

/// Operations of `SimilarityMatrix2<T>` that go through an angle.
#[generate_trait]
pub impl SimilarityMatrix2AngleImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
> of SimilarityMatrix2AngleTrait<T> {
    /// The similarity from a translation, a rotation angle (radians, `Rotation2::new`) and a
    /// nonzero scale. Panics with `nalgebra: zero scale` on zero. Upstream:
    /// `SimilarityMatrix2::new`.
    #[inline(always)]
    fn new(translation: Vector2<T>, angle: T, scaling: T) -> SimilarityMatrix2<T> {
        SimilarityMatrix2Trait::from_parts(
            Translation2 { vector: translation }, Rotation2AngleTrait::new(angle), scaling,
        )
    }
}

/// `a * b`, `b` applied first: translation `a.t + a.scaling * (a.rotation · b.t)`, rotation
/// `a.rotation · b.rotation`, scale `a.scaling · b.scaling`. Upstream: `Mul<Similarity>`.
pub impl SimilarityMatrix2Mul<
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
> of Mul<SimilarityMatrix2<T>> {
    fn mul(lhs: SimilarityMatrix2<T>, rhs: SimilarityMatrix2<T>) -> SimilarityMatrix2<T> {
        let mut res = lhs.mul_isometry(rhs.isometry);
        res.scaling = res.scaling * rhs.scaling;
        res
    }
}

/// `a / b = a * b⁻¹` (upstream's formula). Upstream: `Div<Similarity>`.
pub impl SimilarityMatrix2Div<
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
> of Div<SimilarityMatrix2<T>> {
    #[inline(always)]
    fn div(lhs: SimilarityMatrix2<T>, rhs: SimilarityMatrix2<T>) -> SimilarityMatrix2<T> {
        lhs * rhs.inverse()
    }
}

/// `sim *= t`: `sim = sim * t`. Upstream: `MulAssign<Translation> for Similarity`.
pub impl SimilarityMatrix2MulAssignTranslation2<
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
> of MulAssign<SimilarityMatrix2<T>, Translation2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: SimilarityMatrix2<T>, rhs: Translation2<T>) {
        self = self.mul_translation(rhs);
    }
}

/// `sim *= iso`: `sim = sim * iso`. Upstream: `MulAssign<Isometry> for Similarity`.
pub impl SimilarityMatrix2MulAssignIsometryMatrix2<
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
> of MulAssign<SimilarityMatrix2<T>, IsometryMatrix2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: SimilarityMatrix2<T>, rhs: IsometryMatrix2<T>) {
        self = self.mul_isometry(rhs);
    }
}

/// `a *= b`: `a = a * b`. Upstream: `MulAssign<Similarity> for Similarity`.
pub impl SimilarityMatrix2MulAssign<
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
> of MulAssign<SimilarityMatrix2<T>, SimilarityMatrix2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: SimilarityMatrix2<T>, rhs: SimilarityMatrix2<T>) {
        self = self * rhs;
    }
}

/// `sim /= iso`: `sim = sim * iso⁻¹`. Upstream: `DivAssign<Isometry> for Similarity`.
pub impl SimilarityMatrix2DivAssignIsometryMatrix2<
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
> of DivAssign<SimilarityMatrix2<T>, IsometryMatrix2<T>> {
    #[inline(always)]
    fn div_assign(ref self: SimilarityMatrix2<T>, rhs: IsometryMatrix2<T>) {
        self = self.div_isometry(rhs);
    }
}

/// `a /= b`: `a = a * b⁻¹`. Upstream: `DivAssign<Similarity> for Similarity`.
pub impl SimilarityMatrix2DivAssign<
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
> of DivAssign<SimilarityMatrix2<T>, SimilarityMatrix2<T>> {
    #[inline(always)]
    fn div_assign(ref self: SimilarityMatrix2<T>, rhs: SimilarityMatrix2<T>) {
        self = self * rhs.inverse();
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Similarity`.
pub impl SimilarityMatrix2Default<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Default<SimilarityMatrix2<T>> {
    #[inline(always)]
    fn default() -> SimilarityMatrix2<T> {
        SimilarityMatrix2 {
            isometry: IsometryMatrix2 {
                rotation: Rotation2 {
                    matrix: Matrix2 {
                        m11: R::one(), m21: R::zero(), m12: R::zero(), m22: R::one(),
                    },
                },
                translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
            },
            scaling: R::one(),
        }
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Similarity`.
pub impl SimilarityMatrix2One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<SimilarityMatrix2<T>> {
    #[inline(always)]
    fn one() -> SimilarityMatrix2<T> {
        SimilarityMatrix2Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @SimilarityMatrix2<T>) -> bool {
        *self == SimilarityMatrix2Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @SimilarityMatrix2<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `sim.into()`: the same similarity with its rotation as a unit complex (the first column,
/// exact). Upstream: `SubsetOf<Similarity2> for SimilarityMatrix2` (`nalgebra::convert`).
pub impl Similarity2FromSimilarityMatrix2<
    T, +Copy<T>, +Drop<T>,
> of Into<SimilarityMatrix2<T>, Similarity2<T>> {
    #[inline(always)]
    fn into(self: SimilarityMatrix2<T>) -> Similarity2<T> {
        Similarity2 { isometry: self.isometry.into(), scaling: self.scaling }
    }
}

/// `sim.into()`: the same similarity with its rotation as a matrix (exact). Upstream:
/// `SubsetOf<SimilarityMatrix2> for Similarity2` (`nalgebra::convert`).
pub impl SimilarityMatrix2FromSimilarity2<
    T, +Copy<T>, +Drop<T>, +Neg<T>,
> of Into<Similarity2<T>, SimilarityMatrix2<T>> {
    #[inline(always)]
    fn into(self: Similarity2<T>) -> SimilarityMatrix2<T> {
        SimilarityMatrix2 { isometry: self.isometry.into(), scaling: self.scaling }
    }
}
