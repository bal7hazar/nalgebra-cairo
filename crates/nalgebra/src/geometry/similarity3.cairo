//! `Similarity3`: a 3D uniform scaling, followed by a rotation, followed by a translation
//! (upstream `nalgebra::Similarity3`, which is `Similarity<T, UnitQuaternion<T>, 3>`).
//!
//! `sim * p = translation + scaling * (rotation * p)`. The scaling factor must be nonzero:
//! constructors and scaling mutators panic with `nalgebra: zero scale` on zero.
//!
//! Numeric contract: quaternion rotations go through the `UnitQuaternion` fused kernels; the final
//! scale-plus-translation is fused into one wide accumulation per component. This preserves
//! upstream's fixed-point-observable order (`rotate`, then `scale`, then `translate`) while
//! avoiding a checked scalar addition after the scale.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::{Real, Transcendental};
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;
use super::isometry3::{Isometry3, Isometry3Trait};
use super::quaternion::{ApproxEqTrait, Quaternion};
use super::translation3::Translation3;
use super::unit_quaternion::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait};


pub mod errors {
    pub const ZERO_SCALING: felt252 = 'nalgebra: zero scale';
}

/// A 3D direct similarity: the uniform scale `scaling`, then `isometry.rotation`, then
/// `isometry.translation`. Field names follow upstream (`Similarity { isometry, scaling }`), with
/// `scaling` public in this port so static code can inspect it without an accessor call.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Similarity3<T> {
    pub isometry: Isometry3<T>,
    pub scaling: T,
}

/// Operations of `Similarity3<T>` that need no trigonometry, over a `Real` scalar.
#[generate_trait]
pub impl Similarity3Impl<
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
    +PartialOrd<T>,
> of Similarity3Trait<T> {
    /// The identity similarity (unit scale, no rotation, no translation). Exact. Upstream:
    /// `Similarity3::identity`.
    #[inline(always)]
    fn identity() -> Similarity3<T> {
        Similarity3 { isometry: Isometry3Trait::identity(), scaling: R::one() }
    }

    /// The similarity from `translation`, `rotation` and nonzero `scaling`. Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `Similarity3::from_parts`.
    #[inline(always)]
    fn from_parts(
        translation: Translation3<T>, rotation: UnitQuaternion<T>, scaling: T,
    ) -> Similarity3<T> {
        Self::from_isometry(Isometry3Trait::from_parts(translation, rotation), scaling)
    }

    /// The similarity from an isometry and nonzero scale. Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `Similarity3::from_isometry`.
    #[inline(always)]
    fn from_isometry(isometry: Isometry3<T>, scaling: T) -> Similarity3<T> {
        if scaling == R::zero() {
            core::panic_with_felt252(errors::ZERO_SCALING);
        }
        Similarity3 { isometry, scaling }
    }

    /// The pure nonzero uniform scale. Panics with `nalgebra: zero scale` on zero.
    /// Upstream: `Similarity3::from_scaling`.
    #[inline(always)]
    fn from_scaling(scaling: T) -> Similarity3<T> {
        Self::from_isometry(Isometry3Trait::identity(), scaling)
    }

    /// The stored scaling factor. Exact. Upstream: `scaling`.
    #[inline(always)]
    fn scaling(self: Similarity3<T>) -> T {
        self.scaling
    }

    /// Sets a new nonzero scale, in place (the isometry is unchanged). Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `set_scaling`.
    #[inline(always)]
    fn set_scaling(ref self: Similarity3<T>, scaling: T) {
        self = Self::from_isometry(self.isometry, scaling);
    }

    /// Applies scale `s` BEFORE `self`: only the stored scale changes. Panics on zero `s`.
    /// Upstream: `prepend_scaling`.
    #[inline(always)]
    fn prepend_scaling(self: Similarity3<T>, s: T) -> Similarity3<T> {
        if s == R::zero() {
            core::panic_with_felt252(errors::ZERO_SCALING);
        }
        Similarity3 { isometry: self.isometry, scaling: self.scaling * s }
    }

    /// Applies scale `s` AFTER `self`: the translation and scale are multiplied by `s`. Panics on
    /// zero `s`. Upstream: `append_scaling`.
    #[inline(always)]
    fn append_scaling(self: Similarity3<T>, s: T) -> Similarity3<T> {
        if s == R::zero() {
            core::panic_with_felt252(errors::ZERO_SCALING);
        }
        Similarity3 {
            isometry: Isometry3 {
                rotation: self.isometry.rotation,
                translation: Translation3 {
                    vector: Vector3 {
                        x: self.isometry.translation.vector.x * s,
                        y: self.isometry.translation.vector.y * s,
                        z: self.isometry.translation.vector.z * s,
                    },
                },
            },
            scaling: self.scaling * s,
        }
    }

    /// The inverse similarity. The inverse scale is computed once, then the inverse isometry
    /// translation is divided by the original scale component-wise. Panics only as scalar division
    /// or negation can. Upstream: `inverse`.
    fn inverse(self: Similarity3<T>) -> Similarity3<T> {
        let inv_iso = self.isometry.inverse();
        Similarity3 {
            isometry: Isometry3 {
                rotation: inv_iso.rotation,
                translation: Translation3 {
                    vector: {
                        let (x, y, z) = R::div3(
                            inv_iso.translation.vector.x,
                            inv_iso.translation.vector.y,
                            inv_iso.translation.vector.z,
                            self.scaling,
                        );
                        Vector3 { x, y, z }
                    },
                },
            },
            scaling: R::div(R::one(), self.scaling),
        }
    }

    /// `self * p = translation + scaling * (rotation · p)`. Panics on overflow. Upstream:
    /// `transform_point` (`sim * p`).
    fn transform_point(self: Similarity3<T>, p: Point3<T>) -> Point3<T> {
        let c = Similarity3InternalTrait::rotate_scale_translate(
            self.isometry.rotation,
            Vector3 { x: p.x, y: p.y, z: p.z },
            self.scaling,
            self.isometry.translation.vector,
        );
        Point3 { x: c.x, y: c.y, z: c.z }
    }

    /// `scaling * (rotation · v)`: a similarity acts on a displacement without translation.
    /// Upstream: `transform_vector`.
    #[inline(always)]
    fn transform_vector(self: Similarity3<T>, v: Vector3<T>) -> Vector3<T> {
        let r = self.isometry.rotation.transform_vector(v);
        Vector3 { x: r.x * self.scaling, y: r.y * self.scaling, z: r.z * self.scaling }
    }

    /// `self⁻¹ * p = rotation⁻¹ · (p - translation) / scaling`, with one exact subtraction
    /// before the inverse rotation and one exact quotient per component. Upstream:
    /// `inverse_transform_point`.
    fn inverse_transform_point(self: Similarity3<T>, p: Point3<T>) -> Point3<T> {
        let c = self.isometry.inverse_transform_point(p);
        {
            let (x, y, z) = R::div3(c.x, c.y, c.z, self.scaling);
            Point3 { x, y, z }
        }
    }

    /// `rotation⁻¹ · v / scaling`, one exact quotient per component. Upstream:
    /// `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: Similarity3<T>, v: Vector3<T>) -> Vector3<T> {
        let c = self.isometry.inverse_transform_vector(v);
        {
            let (x, y, z) = R::div3(c.x, c.y, c.z, self.scaling);
            Vector3 { x, y, z }
        }
    }

    /// `Translation(t) ∘ self`: the translation shifts exactly; scale and rotation are unchanged.
    /// Upstream: `append_translation_mut`.
    #[inline(always)]
    fn append_translation_mut(ref self: Similarity3<T>, t: Translation3<T>) {
        let mut isometry = self.isometry;
        isometry.append_translation_mut(t);
        self = Similarity3 { isometry, scaling: self.scaling };
    }

    /// `self ∘ Translation(t)`: the translation shifts by `scaling * (rotation · t)`. Upstream:
    /// `Mul<Translation3>` for similarities.
    fn mul_translation(self: Similarity3<T>, t: Translation3<T>) -> Similarity3<T> {
        Similarity3 {
            isometry: Isometry3 {
                rotation: self.isometry.rotation,
                translation: Translation3 {
                    vector: Similarity3InternalTrait::rotate_scale_translate(
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
    fn append_rotation_mut(ref self: Similarity3<T>, r: UnitQuaternion<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_mut(r);
        self = Similarity3 { isometry, scaling: self.scaling };
    }

    /// `self ∘ Rotation(r)`: rotation before the similarity; translation and scale are unchanged.
    /// Upstream: `Mul<UnitQuaternion>`.
    #[inline(always)]
    fn mul_unit_quaternion(self: Similarity3<T>, r: UnitQuaternion<T>) -> Similarity3<T> {
        Similarity3 { isometry: self.isometry.mul_unit_quaternion(r), scaling: self.scaling }
    }

    /// Appends a rotation about point `p`; the scale is unchanged. Upstream:
    /// `append_rotation_wrt_point_mut`.
    fn append_rotation_wrt_point_mut(ref self: Similarity3<T>, r: UnitQuaternion<T>, p: Point3<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_wrt_point_mut(r, p);
        self = Similarity3 { isometry, scaling: self.scaling };
    }

    /// Appends a rotation about the similarity centre; the translation and scale are unchanged.
    /// Upstream: `append_rotation_wrt_center_mut`.
    #[inline(always)]
    fn append_rotation_wrt_center_mut(ref self: Similarity3<T>, r: UnitQuaternion<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_wrt_center_mut(r);
        self = Similarity3 { isometry, scaling: self.scaling };
    }

    /// The homogeneous matrix with `scaling * rotation_matrix` in the 3x3 block and translation in
    /// the last column. Upstream: `to_homogeneous`.
    fn to_homogeneous(self: Similarity3<T>) -> Matrix4<T> {
        let m = self.isometry.rotation.to_rotation_matrix().matrix;
        Matrix4 {
            m11: m.m11 * self.scaling,
            m21: m.m21 * self.scaling,
            m31: m.m31 * self.scaling,
            m41: R::zero(),
            m12: m.m12 * self.scaling,
            m22: m.m22 * self.scaling,
            m32: m.m32 * self.scaling,
            m42: R::zero(),
            m13: m.m13 * self.scaling,
            m23: m.m23 * self.scaling,
            m33: m.m33 * self.scaling,
            m43: R::zero(),
            m14: self.isometry.translation.vector.x,
            m24: self.isometry.translation.vector.y,
            m34: self.isometry.translation.vector.z,
            m44: R::one(),
        }
    }

    /// Component-wise absolute-difference equality of the isometry and scale in raw ulp. Upstream:
    /// `AbsDiffEq::abs_diff_eq`.
    #[inline(always)]
    fn abs_diff_eq(self: Similarity3<T>, other: Similarity3<T>, ulps: u64) -> bool {
        self.isometry.abs_diff_eq(other.isometry, ulps)
            && R::abs_diff_eq(self.scaling, other.scaling, ulps)
    }

    // --- P09b completion ---------------------------------------------------------------------

    /// The rotation `r` about the point `p`, with the scale `scaling`: translation `r · (-p) + p`
    /// (upstream's formula). Panics with `nalgebra: zero scale` on a zero scale. Upstream:
    /// `Similarity::rotation_wrt_point`.
    #[inline(always)]
    fn rotation_wrt_point(r: UnitQuaternion<T>, p: Point3<T>, scaling: T) -> Similarity3<T> {
        Self::from_isometry(Isometry3Trait::rotation_wrt_point(r, p), scaling)
    }

    /// `Isometry3::face_towards(eye, target, up)` with the scale `scaling`. Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `Similarity3::face_towards`.
    #[inline(always)]
    fn face_towards(
        eye: Point3<T>, target: Point3<T>, up: Vector3<T>, scaling: T,
    ) -> Similarity3<T> {
        Self::from_isometry(Isometry3Trait::face_towards(eye, target, up), scaling)
    }

    /// Deprecated alias of `face_towards`. Upstream: `Similarity3::new_observer_frames`.
    #[inline(always)]
    fn new_observer_frames(
        eye: Point3<T>, target: Point3<T>, up: Vector3<T>, scaling: T,
    ) -> Similarity3<T> {
        Self::face_towards(eye, target, up, scaling)
    }

    /// `Isometry3::look_at_rh(eye, target, up)` with the scale `scaling`. Upstream:
    /// `Similarity3::look_at_rh`.
    #[inline(always)]
    fn look_at_rh(eye: Point3<T>, target: Point3<T>, up: Vector3<T>, scaling: T) -> Similarity3<T> {
        Self::from_isometry(Isometry3Trait::look_at_rh(eye, target, up), scaling)
    }

    /// `Isometry3::look_at_lh(eye, target, up)` with the scale `scaling`. Upstream:
    /// `Similarity3::look_at_lh`.
    #[inline(always)]
    fn look_at_lh(eye: Point3<T>, target: Point3<T>, up: Vector3<T>, scaling: T) -> Similarity3<T> {
        Self::from_isometry(Isometry3Trait::look_at_lh(eye, target, up), scaling)
    }

    /// `self / r = self * r⁻¹`: rotation `rotation / r`, the translation and scale unchanged.
    /// Upstream: `Div<UnitQuaternion> for Similarity3` (a named method: Cairo's `Div` is
    /// homogeneous).
    #[inline(always)]
    fn div_unit_quaternion(self: Similarity3<T>, r: UnitQuaternion<T>) -> Similarity3<T> {
        Similarity3 { isometry: self.isometry.div_unit_quaternion(r), scaling: self.scaling }
    }

    /// `self * iso`: translation `translation + scaling * (rotation · iso.translation)`, rotation
    /// `rotation · iso.rotation`, the same scale. Upstream: `Mul<Isometry> for Similarity`.
    #[inline(always)]
    fn mul_isometry(self: Similarity3<T>, iso: Isometry3<T>) -> Similarity3<T> {
        Similarity3 {
            isometry: Isometry3 {
                rotation: self.isometry.rotation * iso.rotation,
                translation: Translation3 {
                    vector: Similarity3InternalTrait::rotate_scale_translate(
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
    fn div_isometry(self: Similarity3<T>, iso: Isometry3<T>) -> Similarity3<T> {
        Self::mul_isometry(self, iso.inverse())
    }

    /// `relative_eq` of the isometries and of the scales. Upstream:
    /// `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: Similarity3<T>, other: Similarity3<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        self.isometry.relative_eq(other.isometry, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.scaling, other.scaling, epsilon, max_relative)
    }

    /// `ulps_eq` of the isometries and of the scales. Upstream: `approx::UlpsEq::ulps_eq`
    /// (DESIGN D3).
    fn ulps_eq(self: Similarity3<T>, other: Similarity3<T>, epsilon: u64, max_ulps: u32) -> bool {
        self.isometry.ulps_eq(other.isometry, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.scaling, other.scaling, epsilon, max_ulps)
    }

    /// The same similarity with every scalar converted by `Into<T, U>` (the identity for the
    /// single scalar `Fixed`). Upstream: `Similarity3::cast` (and `SubsetOf<Similarity>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Similarity3<T>) -> Similarity3<U> {
        Similarity3 { isometry: self.isometry.cast(), scaling: self.scaling.into() }
    }
}

/// Crate-internal kernels of `Similarity3<T>` (WP 8.0: the public API is strictly upstream's): the
/// fused scale-plus-translation behind `transform_point`, `*` and `mul_translation`.
#[generate_trait]
pub(crate) impl Similarity3InternalImpl<
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
    +PartialOrd<T>,
> of Similarity3InternalTrait<T> {
    /// `scaling * v + t`, fused as one product plus one add in the wide accumulator. This preserves
    /// upstream's `rotate`-then-`scale` rounding when `v` is already rotated. Panics on overflow.
    #[inline(always)]
    fn scale_translate(v: Vector3<T>, scaling: T, t: Vector3<T>) -> Vector3<T> {
        Vector3 {
            x: R::wide_rescale(R::wide_add(R::wide_add_prod(R::wide_zero(), v.x, scaling), t.x)),
            y: R::wide_rescale(R::wide_add(R::wide_add_prod(R::wide_zero(), v.y, scaling), t.y)),
            z: R::wide_rescale(R::wide_add(R::wide_add_prod(R::wide_zero(), v.z, scaling), t.z)),
        }
    }
    /// `scaling * (rotation · v) + t`: a quaternion rotation, then a fused
    /// scale-plus-translation per component. Upstream writes this as `translation *
    /// (rotation * point * scaling)`.
    fn rotate_scale_translate(
        rotation: UnitQuaternion<T>, v: Vector3<T>, scaling: T, t: Vector3<T>,
    ) -> Vector3<T> {
        Self::scale_translate(rotation.transform_vector(v), scaling, t)
    }
}

/// Operations of `Similarity3<T>` that go through an axis-angle vector, hence their own trait.
#[generate_trait]
pub impl Similarity3AngleImpl<
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
    +PartialOrd<T>,
> of Similarity3AngleTrait<T> {
    /// The similarity from translation vector, scaled axis-angle and nonzero scale: one
    /// `UnitQuaternion::from_scaled_axis`. Panics with `nalgebra: zero scale` on zero
    /// scale. Upstream: `Similarity3::new`.
    fn new(translation: Vector3<T>, axisangle: Vector3<T>, scaling: T) -> Similarity3<T> {
        Similarity3Trait::from_parts(
            Translation3 { vector: translation },
            UnitQuaternionAngleTrait::from_scaled_axis(axisangle),
            scaling,
        )
    }
}

/// `a * b`: composition of two similarities, with `b` applied first. Translation is
/// `a.t + a.scaling * (a.rotation · b.t)`, rotation composes, scale multiplies. Upstream: `Mul`.
pub impl Similarity3Mul<
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
    +PartialOrd<T>,
> of Mul<Similarity3<T>> {
    fn mul(lhs: Similarity3<T>, rhs: Similarity3<T>) -> Similarity3<T> {
        Similarity3 {
            isometry: Isometry3 {
                rotation: lhs.isometry.rotation * rhs.isometry.rotation,
                translation: Translation3 {
                    vector: Similarity3InternalTrait::rotate_scale_translate(
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
pub impl Similarity3Div<
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
    +PartialOrd<T>,
> of Div<Similarity3<T>> {
    #[inline(always)]
    fn div(lhs: Similarity3<T>, rhs: Similarity3<T>) -> Similarity3<T> {
        lhs * rhs.inverse()
    }
}

/// `sim *= t`: `sim = sim * t`. Upstream: `MulAssign<Translation> for Similarity`.
pub impl Similarity3MulAssignTranslation3<
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
    +PartialOrd<T>,
> of MulAssign<Similarity3<T>, Translation3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Similarity3<T>, rhs: Translation3<T>) {
        self = self.mul_translation(rhs);
    }
}

/// `sim *= iso`: `sim = sim * iso`. Upstream: `MulAssign<Isometry> for Similarity`.
pub impl Similarity3MulAssignIsometry3<
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
    +PartialOrd<T>,
> of MulAssign<Similarity3<T>, Isometry3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Similarity3<T>, rhs: Isometry3<T>) {
        self = self.mul_isometry(rhs);
    }
}

/// `a *= b`: `a = a * b`. Upstream: `MulAssign<Similarity> for Similarity`.
pub impl Similarity3MulAssign<
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
    +PartialOrd<T>,
> of MulAssign<Similarity3<T>, Similarity3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Similarity3<T>, rhs: Similarity3<T>) {
        self = self * rhs;
    }
}

/// `sim /= iso`: `sim = sim * iso⁻¹`. Upstream: `DivAssign<Isometry> for Similarity`.
pub impl Similarity3DivAssignIsometry3<
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
    +PartialOrd<T>,
> of DivAssign<Similarity3<T>, Isometry3<T>> {
    #[inline(always)]
    fn div_assign(ref self: Similarity3<T>, rhs: Isometry3<T>) {
        self = self.div_isometry(rhs);
    }
}

/// `a /= b`: `a = a * b⁻¹`. Upstream: `DivAssign<Similarity> for Similarity`.
pub impl Similarity3DivAssign<
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
    +PartialOrd<T>,
> of DivAssign<Similarity3<T>, Similarity3<T>> {
    #[inline(always)]
    fn div_assign(ref self: Similarity3<T>, rhs: Similarity3<T>) {
        self = self * rhs.inverse();
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Similarity`.
pub impl Similarity3Default<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Default<Similarity3<T>> {
    #[inline(always)]
    fn default() -> Similarity3<T> {
        let o = R::zero();
        Similarity3 {
            isometry: Isometry3 {
                rotation: UnitQuaternion {
                    quaternion: Quaternion { i: o, j: o, k: o, w: R::one() },
                },
                translation: Translation3 { vector: Vector3 { x: o, y: o, z: o } },
            },
            scaling: R::one(),
        }
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Similarity`.
pub impl Similarity3One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Similarity3<T>> {
    #[inline(always)]
    fn one() -> Similarity3<T> {
        Similarity3Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @Similarity3<T>) -> bool {
        *self == Similarity3Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @Similarity3<T>) -> bool {
        !Self::is_one(self)
    }
}
