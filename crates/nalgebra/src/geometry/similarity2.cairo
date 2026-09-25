//! `Similarity2`: a 2D uniform scaling, followed by a rotation, followed by a translation
//! (upstream `nalgebra::Similarity2`, which is `Similarity<T, UnitComplex<T>, 2>`).
//!
//! `sim * p = translation + scaling * (rotation * p)`. The scaling factor must be nonzero:
//! constructors and scaling mutators panic with `nalgebra: zero scale` on zero.
//!
//! Numeric contract: rotations go through the `UnitComplex` fused kernels; the final
//! scale-plus-translation is fused into one wide accumulation per component. This preserves
//! upstream's fixed-point-observable order (`rotate`, then `scale`, then `translate`) while
//! avoiding a checked scalar addition after the scale.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;
use super::isometry2::{Isometry2, Isometry2Trait};
use super::quaternion::ApproxEqTrait;
use super::translation2::Translation2;
use super::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};


pub mod errors {
    pub const ZERO_SCALING: felt252 = 'nalgebra: zero scale';
}

/// A 2D direct similarity: the uniform scale `scaling`, then `isometry.rotation`, then
/// `isometry.translation`. Field names follow upstream (`Similarity { isometry, scaling }`), with
/// `scaling` public in this port so static code can inspect it without an accessor call.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Similarity2<T> {
    pub isometry: Isometry2<T>,
    pub scaling: T,
}

/// Operations of `Similarity2<T>` that need no trigonometry, over a `Real` scalar.
#[generate_trait]
pub impl Similarity2Impl<
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
> of Similarity2Trait<T> {
    /// The identity similarity (unit scale, no rotation, no translation). Exact. Upstream:
    /// `Similarity2::identity`.
    #[inline(always)]
    fn identity() -> Similarity2<T> {
        Similarity2 { isometry: Isometry2Trait::identity(), scaling: R::one() }
    }

    /// The similarity from `translation`, `rotation` and nonzero `scaling`. Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `Similarity2::from_parts`.
    #[inline(always)]
    fn from_parts(
        translation: Translation2<T>, rotation: UnitComplex<T>, scaling: T,
    ) -> Similarity2<T> {
        Self::from_isometry(Isometry2Trait::from_parts(translation, rotation), scaling)
    }

    /// The similarity from an isometry and nonzero scale. Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `Similarity2::from_isometry`.
    #[inline(always)]
    fn from_isometry(isometry: Isometry2<T>, scaling: T) -> Similarity2<T> {
        if scaling == R::zero() {
            core::panic_with_felt252(errors::ZERO_SCALING);
        }
        Similarity2 { isometry, scaling }
    }

    /// The pure nonzero uniform scale. Panics with `nalgebra: zero scale` on zero.
    /// Upstream: `Similarity2::from_scaling`.
    #[inline(always)]
    fn from_scaling(scaling: T) -> Similarity2<T> {
        Self::from_isometry(Isometry2Trait::identity(), scaling)
    }

    /// The stored scaling factor. Exact. Upstream: `scaling`.
    #[inline(always)]
    fn scaling(self: Similarity2<T>) -> T {
        self.scaling
    }

    /// Sets a new nonzero scale, in place (the isometry is unchanged). Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `set_scaling`.
    #[inline(always)]
    fn set_scaling(ref self: Similarity2<T>, scaling: T) {
        self = Self::from_isometry(self.isometry, scaling);
    }

    /// Applies scale `s` BEFORE `self`: only the stored scale changes. Panics on zero `s`.
    /// Upstream: `prepend_scaling`.
    #[inline(always)]
    fn prepend_scaling(self: Similarity2<T>, s: T) -> Similarity2<T> {
        if s == R::zero() {
            core::panic_with_felt252(errors::ZERO_SCALING);
        }
        Similarity2 { isometry: self.isometry, scaling: self.scaling * s }
    }

    /// Applies scale `s` AFTER `self`: the translation and scale are multiplied by `s`. Panics on
    /// zero `s`. Upstream: `append_scaling`.
    #[inline(always)]
    fn append_scaling(self: Similarity2<T>, s: T) -> Similarity2<T> {
        if s == R::zero() {
            core::panic_with_felt252(errors::ZERO_SCALING);
        }
        Similarity2 {
            isometry: Isometry2 {
                rotation: self.isometry.rotation,
                translation: Translation2 {
                    vector: Vector2 {
                        x: self.isometry.translation.vector.x * s,
                        y: self.isometry.translation.vector.y * s,
                    },
                },
            },
            scaling: self.scaling * s,
        }
    }

    /// The inverse similarity. The inverse scale is computed once, then the inverse isometry
    /// translation is divided by the original scale component-wise. Panics only as scalar division
    /// or negation can. Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: Similarity2<T>) -> Similarity2<T> {
        let inv_iso = self.isometry.inverse();
        Similarity2 {
            isometry: Isometry2 {
                rotation: inv_iso.rotation,
                translation: Translation2 {
                    vector: Vector2 {
                        x: R::div(inv_iso.translation.vector.x, self.scaling),
                        y: R::div(inv_iso.translation.vector.y, self.scaling),
                    },
                },
            },
            scaling: R::div(R::one(), self.scaling),
        }
    }

    /// `self = self.inverse()` in place: the by-value form is the cheapest (a rotation and a
    /// translation to rebuild, nothing to reuse), so the bits are those of `inverse`. Panics as
    /// `inverse` does. Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: Similarity2<T>) {
        self = Self::inverse(self);
    }

    /// `self = self.prepend_scaling(s)` in place: only the stored scale is multiplied by `s`.
    /// Panics with `nalgebra: zero scale` on a zero `s`, and on overflow. Upstream:
    /// `prepend_scaling_mut`.
    #[inline(always)]
    fn prepend_scaling_mut(ref self: Similarity2<T>, s: T) {
        self = Self::prepend_scaling(self, s);
    }

    /// `self = self.append_scaling(s)` in place: the translation and the scale are multiplied by
    /// `s`. Panics with `nalgebra: zero scale` on a zero `s`, and on overflow. Upstream:
    /// `append_scaling_mut`.
    #[inline(always)]
    fn append_scaling_mut(ref self: Similarity2<T>, s: T) {
        self = Self::append_scaling(self, s);
    }

    /// `self * p = translation + scaling * (rotation · p)`. Panics on overflow. Upstream:
    /// `transform_point` (`sim * p`).
    #[inline(always)]
    fn transform_point(self: Similarity2<T>, p: Point2<T>) -> Point2<T> {
        let c = Similarity2InternalTrait::rotate_scale_translate(
            self.isometry.rotation,
            Vector2 { x: p.x, y: p.y },
            self.scaling,
            self.isometry.translation.vector,
        );
        Point2 { x: c.x, y: c.y }
    }

    /// `scaling * (rotation · v)`: a similarity acts on a displacement without translation.
    /// Upstream: `transform_vector`.
    #[inline(always)]
    fn transform_vector(self: Similarity2<T>, v: Vector2<T>) -> Vector2<T> {
        let r = self.isometry.rotation.transform_vector(v);
        Vector2 { x: r.x * self.scaling, y: r.y * self.scaling }
    }

    /// `self⁻¹ * p = rotation⁻¹ · (p - translation) / scaling`, with one exact subtraction
    /// before the inverse rotation and one exact quotient per component. Upstream:
    /// `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Similarity2<T>, p: Point2<T>) -> Point2<T> {
        let c = self.isometry.inverse_transform_point(p);
        Point2 { x: R::div(c.x, self.scaling), y: R::div(c.y, self.scaling) }
    }

    /// `rotation⁻¹ · v / scaling`, one exact quotient per component. Upstream:
    /// `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: Similarity2<T>, v: Vector2<T>) -> Vector2<T> {
        let c = self.isometry.inverse_transform_vector(v);
        Vector2 { x: R::div(c.x, self.scaling), y: R::div(c.y, self.scaling) }
    }

    /// `Translation(t) ∘ self`: the translation shifts exactly; scale and rotation are unchanged.
    /// Upstream: `append_translation_mut`.
    #[inline(always)]
    fn append_translation_mut(ref self: Similarity2<T>, t: Translation2<T>) {
        let mut isometry = self.isometry;
        isometry.append_translation_mut(t);
        self = Similarity2 { isometry, scaling: self.scaling };
    }

    /// `self ∘ Translation(t)`: the translation shifts by `scaling * (rotation · t)`. Upstream:
    /// `Mul<Translation2>` for similarities.
    #[inline(always)]
    fn mul_translation(self: Similarity2<T>, t: Translation2<T>) -> Similarity2<T> {
        Similarity2 {
            isometry: Isometry2 {
                rotation: self.isometry.rotation,
                translation: Translation2 {
                    vector: Similarity2InternalTrait::rotate_scale_translate(
                        self.isometry.rotation,
                        t.vector,
                        self.scaling,
                        self.isometry.translation.vector,
                    ),
                },
            },
            scaling: self.scaling,
        }
    }

    /// `Rotation(r) ∘ self`: the isometry part handles the rotation about the origin; scale is
    /// unchanged. Upstream: `append_rotation_mut`.
    #[inline(always)]
    fn append_rotation_mut(ref self: Similarity2<T>, r: UnitComplex<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_mut(r);
        self = Similarity2 { isometry, scaling: self.scaling };
    }

    /// `self ∘ Rotation(r)`: rotation before the similarity; translation and scale are unchanged.
    /// Upstream: `Mul<UnitComplex>`.
    #[inline(always)]
    fn mul_unit_complex(self: Similarity2<T>, r: UnitComplex<T>) -> Similarity2<T> {
        Similarity2 { isometry: self.isometry.mul_unit_complex(r), scaling: self.scaling }
    }

    /// Appends a rotation about point `p`; the scale is unchanged. Upstream:
    /// `append_rotation_wrt_point_mut`.
    #[inline(always)]
    fn append_rotation_wrt_point_mut(ref self: Similarity2<T>, r: UnitComplex<T>, p: Point2<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_wrt_point_mut(r, p);
        self = Similarity2 { isometry, scaling: self.scaling };
    }

    /// Appends a rotation about the similarity centre; the translation and scale are unchanged.
    /// Upstream: `append_rotation_wrt_center_mut`.
    #[inline(always)]
    fn append_rotation_wrt_center_mut(ref self: Similarity2<T>, r: UnitComplex<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_wrt_center_mut(r);
        self = Similarity2 { isometry, scaling: self.scaling };
    }

    /// The homogeneous matrix `[[s*re, -s*im, tx], [s*im, s*re, ty], [0, 0, 1]]`, with one product
    /// per scaled rotation entry. Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Similarity2<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.isometry.rotation.re * self.scaling,
            m21: self.isometry.rotation.im * self.scaling,
            m31: R::zero(),
            m12: -self.isometry.rotation.im * self.scaling,
            m22: self.isometry.rotation.re * self.scaling,
            m32: R::zero(),
            m13: self.isometry.translation.vector.x,
            m23: self.isometry.translation.vector.y,
            m33: R::one(),
        }
    }

    /// Component-wise absolute-difference equality of the isometry and scale in raw ulp. Upstream:
    /// `AbsDiffEq::abs_diff_eq`.
    #[inline(always)]
    fn abs_diff_eq(self: Similarity2<T>, other: Similarity2<T>, ulps: u64) -> bool {
        self.isometry.abs_diff_eq(other.isometry, ulps)
            && R::abs_diff_eq(self.scaling, other.scaling, ulps)
    }

    // --- P09b completion ---------------------------------------------------------------------

    /// The rotation `r` about the point `p`, with the scale `scaling`: translation `r · (-p) + p`
    /// (upstream's formula, one fused kernel per component). Panics with `nalgebra: zero scale`
    /// on a zero scale. Upstream: `Similarity::rotation_wrt_point`.
    #[inline(always)]
    fn rotation_wrt_point(r: UnitComplex<T>, p: Point2<T>, scaling: T) -> Similarity2<T> {
        Self::from_isometry(Isometry2Trait::rotation_wrt_point(r, p), scaling)
    }

    /// `self / r = self * r⁻¹`: rotation `rotation / r`, the translation and scale unchanged.
    /// Upstream: `Div<UnitComplex> for Similarity2` (a named method: Cairo's `Div` is
    /// homogeneous).
    #[inline(always)]
    fn div_unit_complex(self: Similarity2<T>, r: UnitComplex<T>) -> Similarity2<T> {
        Similarity2 { isometry: self.isometry.div_unit_complex(r), scaling: self.scaling }
    }

    /// `self * iso`: translation `translation + scaling * (rotation · iso.translation)`, rotation
    /// `rotation · iso.rotation`, the same scale. Upstream: `Mul<Isometry> for Similarity`.
    #[inline(always)]
    fn mul_isometry(self: Similarity2<T>, iso: Isometry2<T>) -> Similarity2<T> {
        Similarity2 {
            isometry: Isometry2 {
                rotation: self.isometry.rotation * iso.rotation,
                translation: Translation2 {
                    vector: Similarity2InternalTrait::rotate_scale_translate(
                        self.isometry.rotation,
                        iso.translation.vector,
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
    fn div_isometry(self: Similarity2<T>, iso: Isometry2<T>) -> Similarity2<T> {
        Self::mul_isometry(self, iso.inverse())
    }

    /// `relative_eq` of the isometries and of the scales. Upstream:
    /// `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Similarity2<T>, other: Similarity2<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        self.isometry.relative_eq(other.isometry, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.scaling, other.scaling, epsilon, max_relative)
    }

    /// `ulps_eq` of the isometries and of the scales. Upstream: `approx::UlpsEq::ulps_eq`
    /// (DESIGN D3).
    fn ulps_eq(self: Similarity2<T>, other: Similarity2<T>, epsilon: u64, max_ulps: u32) -> bool {
        self.isometry.ulps_eq(other.isometry, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.scaling, other.scaling, epsilon, max_ulps)
    }

    /// The same similarity with every scalar converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `Similarity2::cast` (and `SubsetOf<Similarity>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Similarity2<T>) -> Similarity2<U> {
        Similarity2 { isometry: self.isometry.cast(), scaling: self.scaling.into() }
    }
}

/// Crate-internal kernels of `Similarity2<T>` (WP 8.0: the public API is strictly upstream's): the
/// fused scale-plus-translation behind `transform_point`, `*` and `mul_translation`.
#[generate_trait]
pub(crate) impl Similarity2InternalImpl<
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
> of Similarity2InternalTrait<T> {
    /// `scaling * v + t`, fused as one product plus one add in the wide accumulator. This preserves
    /// upstream's `rotate`-then-`scale` rounding when `v` is already rotated. Panics on overflow.
    #[inline(always)]
    fn scale_translate(v: Vector2<T>, scaling: T, t: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: R::wide_rescale(R::wide_add(R::wide_add_prod(R::wide_zero(), v.x, scaling), t.x)),
            y: R::wide_rescale(R::wide_add(R::wide_add_prod(R::wide_zero(), v.y, scaling), t.y)),
        }
    }
    /// `scaling * (rotation · v) + t`: two fused rotation components, then a fused
    /// scale-plus-translation per component. Upstream writes this as `translation *
    /// (rotation * point * scaling)`.
    #[inline(always)]
    fn rotate_scale_translate(
        rotation: UnitComplex<T>, v: Vector2<T>, scaling: T, t: Vector2<T>,
    ) -> Vector2<T> {
        Self::scale_translate(rotation.transform_vector(v), scaling, t)
    }
}

/// Operations of `Similarity2<T>` that go through an angle, hence their own trait.
#[generate_trait]
pub impl Similarity2AngleImpl<
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
> of Similarity2AngleTrait<T> {
    /// The similarity from translation vector, rotation angle in radians and nonzero scale: one
    /// `sin_cos`. Panics with `nalgebra: zero scale` on zero scale. Upstream:
    /// `Similarity2::new`.
    #[inline(always)]
    fn new(translation: Vector2<T>, angle: T, scaling: T) -> Similarity2<T> {
        Similarity2Trait::from_parts(
            Translation2 { vector: translation }, UnitComplexAngleTrait::new(angle), scaling,
        )
    }
}

/// `a * b`: composition of two similarities, with `b` applied first. Translation is
/// `a.t + a.scaling * (a.rotation · b.t)`, rotation composes, scale multiplies. Upstream: `Mul`.
pub impl Similarity2Mul<
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
> of Mul<Similarity2<T>> {
    fn mul(lhs: Similarity2<T>, rhs: Similarity2<T>) -> Similarity2<T> {
        Similarity2 {
            isometry: Isometry2 {
                rotation: lhs.isometry.rotation * rhs.isometry.rotation,
                translation: Translation2 {
                    vector: Similarity2InternalTrait::rotate_scale_translate(
                        lhs.isometry.rotation,
                        rhs.isometry.translation.vector,
                        lhs.scaling,
                        lhs.isometry.translation.vector,
                    ),
                },
            },
            scaling: lhs.scaling * rhs.scaling,
        }
    }
}

/// `a / b = a * b⁻¹` (upstream's formula: the inverse is materialised). Upstream:
/// `Div<Similarity>`.
pub impl Similarity2Div<
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
> of Div<Similarity2<T>> {
    #[inline(always)]
    fn div(lhs: Similarity2<T>, rhs: Similarity2<T>) -> Similarity2<T> {
        lhs * rhs.inverse()
    }
}

/// `sim *= t`: `sim = sim * t`. Upstream: `MulAssign<Translation> for Similarity`.
pub impl Similarity2MulAssignTranslation2<
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
> of MulAssign<Similarity2<T>, Translation2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Similarity2<T>, rhs: Translation2<T>) {
        self = self.mul_translation(rhs);
    }
}

/// `sim *= iso`: `sim = sim * iso`. Upstream: `MulAssign<Isometry> for Similarity`.
pub impl Similarity2MulAssignIsometry2<
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
> of MulAssign<Similarity2<T>, Isometry2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Similarity2<T>, rhs: Isometry2<T>) {
        self = self.mul_isometry(rhs);
    }
}

/// `a *= b`: `a = a * b`. Upstream: `MulAssign<Similarity> for Similarity`.
pub impl Similarity2MulAssign<
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
> of MulAssign<Similarity2<T>, Similarity2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Similarity2<T>, rhs: Similarity2<T>) {
        self = self * rhs;
    }
}

/// `sim /= iso`: `sim = sim * iso⁻¹`. Upstream: `DivAssign<Isometry> for Similarity`.
pub impl Similarity2DivAssignIsometry2<
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
> of DivAssign<Similarity2<T>, Isometry2<T>> {
    #[inline(always)]
    fn div_assign(ref self: Similarity2<T>, rhs: Isometry2<T>) {
        self = self.div_isometry(rhs);
    }
}

/// `a /= b`: `a = a * b⁻¹`. Upstream: `DivAssign<Similarity> for Similarity`.
pub impl Similarity2DivAssign<
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
> of DivAssign<Similarity2<T>, Similarity2<T>> {
    #[inline(always)]
    fn div_assign(ref self: Similarity2<T>, rhs: Similarity2<T>) {
        self = self * rhs.inverse();
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Similarity`.
pub impl Similarity2Default<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Default<Similarity2<T>> {
    #[inline(always)]
    fn default() -> Similarity2<T> {
        Similarity2 {
            isometry: Isometry2 {
                rotation: UnitComplex { re: R::one(), im: R::zero() },
                translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
            },
            scaling: R::one(),
        }
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Similarity`.
pub impl Similarity2One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Similarity2<T>> {
    #[inline(always)]
    fn one() -> Similarity2<T> {
        Similarity2Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @Similarity2<T>) -> bool {
        *self == Similarity2Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @Similarity2<T>) -> bool {
        !Self::is_one(self)
    }
}
