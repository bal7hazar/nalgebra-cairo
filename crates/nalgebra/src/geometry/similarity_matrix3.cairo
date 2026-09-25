//! `SimilarityMatrix3`: a 3D uniform scaling, followed by a rotation stored as a `Rotation3`
//! MATRIX, followed by a translation (upstream `nalgebra::SimilarityMatrix3`, which is
//! `Similarity<T, Rotation3<T>, 3>`).
//!
//! `sim * p = translation + scaling * (rotation * p)`. The rotation-matrix instance of upstream's
//! generic `Similarity` is a struct of its own, with the method set, field names and semantics of
//! `Similarity3` (whose rotation is a `UnitQuaternion`); the two convert into each other with
//! `.into()` (Shepperd's method one way, upstream's quaternion-to-matrix form the other). The
//! scaling factor must be nonzero: constructors and scaling mutators panic with `nalgebra: zero
//! scale` on zero (`similarity3::errors::ZERO_SCALING`).
//!
//! Numeric contract: the rotations go through the `Rotation3` fused kernels, the final
//! scale-plus-translation is one wide accumulation per component (`scale_translate`, shared with
//! `Similarity3`), preserving upstream's order (rotate, then scale, then translate).

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::Matrix3;
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;
use super::isometry_matrix3::{IsometryMatrix3, IsometryMatrix3Trait};
use super::quaternion::ApproxEqTrait;
use super::rotation3::{Rotation3, Rotation3AngleTrait, Rotation3Trait};
use super::similarity3::errors::ZERO_SCALING;
use super::similarity3::{Similarity3, Similarity3InternalTrait};
use super::translation3::Translation3;

/// A 3D direct similarity whose rotation is a matrix: the uniform scale `scaling`, then
/// `isometry.rotation`, then `isometry.translation` (upstream's fields).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct SimilarityMatrix3<T> {
    pub isometry: IsometryMatrix3<T>,
    pub scaling: T,
}

/// Operations of `SimilarityMatrix3<T>` that need no trigonometry, over a `Real` scalar.
#[generate_trait]
pub impl SimilarityMatrix3Impl<
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
> of SimilarityMatrix3Trait<T> {
    /// The identity similarity (unit scale, no rotation, no translation). Exact. Upstream:
    /// `Similarity::identity`.
    #[inline(always)]
    fn identity() -> SimilarityMatrix3<T> {
        SimilarityMatrix3 { isometry: IsometryMatrix3Trait::identity(), scaling: R::one() }
    }

    /// The similarity from `translation`, `rotation` and nonzero `scaling`. Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `Similarity::from_parts`.
    #[inline(always)]
    fn from_parts(
        translation: Translation3<T>, rotation: Rotation3<T>, scaling: T,
    ) -> SimilarityMatrix3<T> {
        Self::from_isometry(IsometryMatrix3Trait::from_parts(translation, rotation), scaling)
    }

    /// The similarity from an isometry and a nonzero scale. Panics with `nalgebra: zero scale` on
    /// zero. Upstream: `Similarity::from_isometry`.
    #[inline(always)]
    fn from_isometry(isometry: IsometryMatrix3<T>, scaling: T) -> SimilarityMatrix3<T> {
        if scaling == R::zero() {
            core::panic_with_felt252(ZERO_SCALING);
        }
        SimilarityMatrix3 { isometry, scaling }
    }

    /// The pure nonzero uniform scale. Panics with `nalgebra: zero scale` on zero. Upstream:
    /// `Similarity::from_scaling`.
    #[inline(always)]
    fn from_scaling(scaling: T) -> SimilarityMatrix3<T> {
        Self::from_isometry(IsometryMatrix3Trait::identity(), scaling)
    }

    /// The rotation `r` about the point `p` followed by nothing else, with the scale `scaling`:
    /// translation `r · (-p) + p` (upstream's formula, one `rotate_translate`). Panics with
    /// `nalgebra: zero scale` on a zero scale. Upstream: `Similarity::rotation_wrt_point`.
    #[inline(always)]
    fn rotation_wrt_point(r: Rotation3<T>, p: Point3<T>, scaling: T) -> SimilarityMatrix3<T> {
        Self::from_isometry(IsometryMatrix3Trait::rotation_wrt_point(r, p), scaling)
    }

    /// `IsometryMatrix3::face_towards(eye, target, up)` with the scale `scaling`. Panics with
    /// `nalgebra: zero scale` on zero. Upstream: `SimilarityMatrix3::face_towards`.
    #[inline(always)]
    fn face_towards(
        eye: Point3<T>, target: Point3<T>, up: Vector3<T>, scaling: T,
    ) -> SimilarityMatrix3<T> {
        Self::from_isometry(IsometryMatrix3Trait::face_towards(eye, target, up), scaling)
    }

    /// Deprecated alias of `face_towards`. Upstream: `SimilarityMatrix3::new_observer_frames`.
    #[inline(always)]
    fn new_observer_frames(
        eye: Point3<T>, target: Point3<T>, up: Vector3<T>, scaling: T,
    ) -> SimilarityMatrix3<T> {
        Self::face_towards(eye, target, up, scaling)
    }

    /// `IsometryMatrix3::look_at_rh(eye, target, up)` with the scale `scaling`. Upstream:
    /// `SimilarityMatrix3::look_at_rh`.
    #[inline(always)]
    fn look_at_rh(
        eye: Point3<T>, target: Point3<T>, up: Vector3<T>, scaling: T,
    ) -> SimilarityMatrix3<T> {
        Self::from_isometry(IsometryMatrix3Trait::look_at_rh(eye, target, up), scaling)
    }

    /// `IsometryMatrix3::look_at_lh(eye, target, up)` with the scale `scaling`. Upstream:
    /// `SimilarityMatrix3::look_at_lh`.
    #[inline(always)]
    fn look_at_lh(
        eye: Point3<T>, target: Point3<T>, up: Vector3<T>, scaling: T,
    ) -> SimilarityMatrix3<T> {
        Self::from_isometry(IsometryMatrix3Trait::look_at_lh(eye, target, up), scaling)
    }

    /// The stored scaling factor. Exact. Upstream: `scaling`.
    #[inline(always)]
    fn scaling(self: SimilarityMatrix3<T>) -> T {
        self.scaling
    }

    /// Sets a new nonzero scale, in place. Panics with `nalgebra: zero scale` on zero. Upstream:
    /// `set_scaling`.
    #[inline(always)]
    fn set_scaling(ref self: SimilarityMatrix3<T>, scaling: T) {
        self = Self::from_isometry(self.isometry, scaling);
    }

    /// Applies the scale `s` BEFORE `self`: only the stored scale changes (one product). Panics
    /// on a zero `s`. Upstream: `prepend_scaling`.
    #[inline(always)]
    fn prepend_scaling(self: SimilarityMatrix3<T>, s: T) -> SimilarityMatrix3<T> {
        if s == R::zero() {
            core::panic_with_felt252(ZERO_SCALING);
        }
        SimilarityMatrix3 { isometry: self.isometry, scaling: self.scaling * s }
    }

    /// Applies the scale `s` AFTER `self`: the translation and the scale are multiplied by `s`.
    /// Panics on a zero `s`. Upstream: `append_scaling`.
    #[inline(always)]
    fn append_scaling(self: SimilarityMatrix3<T>, s: T) -> SimilarityMatrix3<T> {
        if s == R::zero() {
            core::panic_with_felt252(ZERO_SCALING);
        }
        let t = self.isometry.translation.vector;
        SimilarityMatrix3 {
            isometry: IsometryMatrix3 {
                rotation: self.isometry.rotation,
                translation: Translation3 {
                    vector: Vector3 { x: t.x * s, y: t.y * s, z: t.z * s },
                },
            },
            scaling: self.scaling * s,
        }
    }

    /// The inverse similarity, like `Similarity3::inverse`: the inverse isometry, its translation
    /// divided by the scale (one correctly rounded quotient per component), the scale inverted.
    /// Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: SimilarityMatrix3<T>) -> SimilarityMatrix3<T> {
        let inv_iso = self.isometry.inverse();
        let t = inv_iso.translation.vector;
        SimilarityMatrix3 {
            isometry: IsometryMatrix3 {
                rotation: inv_iso.rotation,
                translation: Translation3 {
                    vector: {
                        let (x, y, z) = R::div3(t.x, t.y, t.z, self.scaling);
                        Vector3 { x, y, z }
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
    fn inverse_mut(ref self: SimilarityMatrix3<T>) {
        self = Self::inverse(self);
    }

    /// `self = self.prepend_scaling(s)` in place: only the stored scale is multiplied by `s`.
    /// Panics with `nalgebra: zero scale` on a zero `s`, and on overflow. Upstream:
    /// `prepend_scaling_mut`.
    #[inline(always)]
    fn prepend_scaling_mut(ref self: SimilarityMatrix3<T>, s: T) {
        self = Self::prepend_scaling(self, s);
    }

    /// `self = self.append_scaling(s)` in place: the translation and the scale are multiplied by
    /// `s`. Panics with `nalgebra: zero scale` on a zero `s`, and on overflow. Upstream:
    /// `append_scaling_mut`.
    #[inline(always)]
    fn append_scaling_mut(ref self: SimilarityMatrix3<T>, s: T) {
        self = Self::append_scaling(self, s);
    }

    /// `self * p = translation + scaling * (rotation · p)`. Upstream: `transform_point`.
    #[inline(always)]
    fn transform_point(self: SimilarityMatrix3<T>, p: Point3<T>) -> Point3<T> {
        let c = Similarity3InternalTrait::scale_translate(
            self.isometry.rotation.transform_vector(Vector3 { x: p.x, y: p.y, z: p.z }),
            self.scaling,
            self.isometry.translation.vector,
        );
        Point3 { x: c.x, y: c.y, z: c.z }
    }

    /// `scaling * (rotation · v)`. Upstream: `transform_vector`.
    #[inline(always)]
    fn transform_vector(self: SimilarityMatrix3<T>, v: Vector3<T>) -> Vector3<T> {
        let r = self.isometry.rotation.transform_vector(v);
        Vector3 { x: r.x * self.scaling, y: r.y * self.scaling, z: r.z * self.scaling }
    }

    /// `self⁻¹ * p = rotationᵀ · (p - translation) / scaling`. Upstream:
    /// `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: SimilarityMatrix3<T>, p: Point3<T>) -> Point3<T> {
        let c = self.isometry.inverse_transform_point(p);
        {
            let (x, y, z) = R::div3(c.x, c.y, c.z, self.scaling);
            Point3 { x, y, z }
        }
    }

    /// `rotationᵀ · v / scaling`. Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: SimilarityMatrix3<T>, v: Vector3<T>) -> Vector3<T> {
        let c = self.isometry.inverse_transform_vector(v);
        {
            let (x, y, z) = R::div3(c.x, c.y, c.z, self.scaling);
            Vector3 { x, y, z }
        }
    }

    /// `Translation(t) ∘ self`, in place. Upstream: `append_translation_mut`.
    #[inline(always)]
    fn append_translation_mut(ref self: SimilarityMatrix3<T>, t: Translation3<T>) {
        let mut isometry = self.isometry;
        isometry.append_translation_mut(t);
        self = SimilarityMatrix3 { isometry, scaling: self.scaling };
    }

    /// `self * t`: the translation shifted by `scaling * (rotation · t)`. Upstream:
    /// `Mul<Translation> for Similarity` (a named method: Cairo's `Mul` is homogeneous).
    #[inline(always)]
    fn mul_translation(self: SimilarityMatrix3<T>, t: Translation3<T>) -> SimilarityMatrix3<T> {
        SimilarityMatrix3 {
            isometry: IsometryMatrix3 {
                rotation: self.isometry.rotation,
                translation: Translation3 {
                    vector: Similarity3InternalTrait::scale_translate(
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
    fn append_rotation_mut(ref self: SimilarityMatrix3<T>, r: Rotation3<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_mut(r);
        self = SimilarityMatrix3 { isometry, scaling: self.scaling };
    }

    /// The rotation `r` appended about the point `p`, in place. Upstream:
    /// `append_rotation_wrt_point_mut`.
    #[inline(always)]
    fn append_rotation_wrt_point_mut(
        ref self: SimilarityMatrix3<T>, r: Rotation3<T>, p: Point3<T>,
    ) {
        let mut isometry = self.isometry;
        isometry.append_rotation_wrt_point_mut(r, p);
        self = SimilarityMatrix3 { isometry, scaling: self.scaling };
    }

    /// The rotation `r` appended about the similarity's centre, in place. Upstream:
    /// `append_rotation_wrt_center_mut`.
    #[inline(always)]
    fn append_rotation_wrt_center_mut(ref self: SimilarityMatrix3<T>, r: Rotation3<T>) {
        let mut isometry = self.isometry;
        isometry.append_rotation_wrt_center_mut(r);
        self = SimilarityMatrix3 { isometry, scaling: self.scaling };
    }

    /// `self * r`: rotation before the similarity. Upstream: `Mul<Rotation> for
    /// Similarity<T, Rotation<T, D>, D>`.
    #[inline(always)]
    fn mul_rotation(self: SimilarityMatrix3<T>, r: Rotation3<T>) -> SimilarityMatrix3<T> {
        SimilarityMatrix3 { isometry: self.isometry.mul_rotation(r), scaling: self.scaling }
    }

    /// `self / r = self * r⁻¹`. Upstream: `Div<Rotation> for Similarity<T, Rotation<T, D>, D>`.
    #[inline(always)]
    fn div_rotation(self: SimilarityMatrix3<T>, r: Rotation3<T>) -> SimilarityMatrix3<T> {
        SimilarityMatrix3 { isometry: self.isometry.div_rotation(r), scaling: self.scaling }
    }

    /// `self * iso`: translation `translation + scaling * (rotation · iso.translation)`, rotation
    /// `rotation · iso.rotation`, the same scale. Upstream: `Mul<Isometry> for Similarity`.
    #[inline(always)]
    fn mul_isometry(self: SimilarityMatrix3<T>, iso: IsometryMatrix3<T>) -> SimilarityMatrix3<T> {
        SimilarityMatrix3 {
            isometry: IsometryMatrix3 {
                rotation: self.isometry.rotation * iso.rotation,
                translation: Translation3 {
                    vector: Similarity3InternalTrait::scale_translate(
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
    fn div_isometry(self: SimilarityMatrix3<T>, iso: IsometryMatrix3<T>) -> SimilarityMatrix3<T> {
        Self::mul_isometry(self, iso.inverse())
    }

    /// The homogeneous matrix: the rotation block times the scale (one product per entry), the
    /// translation, then `(0, 0, 0, 1)`. Upstream: `to_homogeneous`.
    fn to_homogeneous(self: SimilarityMatrix3<T>) -> Matrix4<T> {
        let (m, s) = (self.isometry.rotation.matrix, self.scaling);
        let t = self.isometry.translation.vector;
        Matrix4 {
            m11: m.m11 * s,
            m21: m.m21 * s,
            m31: m.m31 * s,
            m41: R::zero(),
            m12: m.m12 * s,
            m22: m.m22 * s,
            m32: m.m32 * s,
            m42: R::zero(),
            m13: m.m13 * s,
            m23: m.m23 * s,
            m33: m.m33 * s,
            m43: R::zero(),
            m14: t.x,
            m24: t.y,
            m34: t.z,
            m44: R::one(),
        }
    }

    /// Component-wise `abs_diff_eq` of the isometry and the scale, in ulp. Upstream:
    /// `AbsDiffEq::abs_diff_eq` (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: SimilarityMatrix3<T>, other: SimilarityMatrix3<T>, ulps: u64) -> bool {
        self.isometry.abs_diff_eq(other.isometry, ulps)
            && R::abs_diff_eq(self.scaling, other.scaling, ulps)
    }

    /// `relative_eq` of the isometries and of the scales. Upstream:
    /// `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(
        self: SimilarityMatrix3<T>, other: SimilarityMatrix3<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        self.isometry.relative_eq(other.isometry, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.scaling, other.scaling, epsilon, max_relative)
    }

    /// `ulps_eq` of the isometries and of the scales. Upstream: `approx::UlpsEq::ulps_eq`
    /// (DESIGN D3).
    fn ulps_eq(
        self: SimilarityMatrix3<T>, other: SimilarityMatrix3<T>, epsilon: u64, max_ulps: u32,
    ) -> bool {
        self.isometry.ulps_eq(other.isometry, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.scaling, other.scaling, epsilon, max_ulps)
    }

    /// The same similarity with every scalar converted by `Into<T, U>`. Upstream:
    /// `SimilarityMatrix3::cast` (and `SubsetOf<Similarity>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: SimilarityMatrix3<T>) -> SimilarityMatrix3<U> {
        SimilarityMatrix3 { isometry: self.isometry.cast(), scaling: self.scaling.into() }
    }
}

/// Operations of `SimilarityMatrix3<T>` that go through an angle.
#[generate_trait]
pub impl SimilarityMatrix3AngleImpl<
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
> of SimilarityMatrix3AngleTrait<T> {
    /// The similarity from a translation, a rotation vector (`Rotation3::new`, the exponential
    /// map) and a nonzero scale. Panics with `nalgebra: zero scale` on zero. Upstream:
    /// `SimilarityMatrix3::new`.
    #[inline(always)]
    fn new(translation: Vector3<T>, axisangle: Vector3<T>, scaling: T) -> SimilarityMatrix3<T> {
        SimilarityMatrix3Trait::from_parts(
            Translation3 { vector: translation }, Rotation3AngleTrait::new(axisangle), scaling,
        )
    }
}

/// `a * b`, `b` applied first: translation `a.t + a.scaling * (a.rotation · b.t)`, rotation
/// `a.rotation · b.rotation`, scale `a.scaling · b.scaling`. Upstream: `Mul<Similarity>`.
pub impl SimilarityMatrix3Mul<
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
> of Mul<SimilarityMatrix3<T>> {
    fn mul(lhs: SimilarityMatrix3<T>, rhs: SimilarityMatrix3<T>) -> SimilarityMatrix3<T> {
        let mut res = lhs.mul_isometry(rhs.isometry);
        res.scaling = res.scaling * rhs.scaling;
        res
    }
}

/// `a / b = a * b⁻¹` (upstream's formula). Upstream: `Div<Similarity>`.
pub impl SimilarityMatrix3Div<
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
> of Div<SimilarityMatrix3<T>> {
    #[inline(always)]
    fn div(lhs: SimilarityMatrix3<T>, rhs: SimilarityMatrix3<T>) -> SimilarityMatrix3<T> {
        lhs * rhs.inverse()
    }
}

/// `sim *= t`: `sim = sim * t`. Upstream: `MulAssign<Translation> for Similarity`.
pub impl SimilarityMatrix3MulAssignTranslation3<
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
> of MulAssign<SimilarityMatrix3<T>, Translation3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: SimilarityMatrix3<T>, rhs: Translation3<T>) {
        self = self.mul_translation(rhs);
    }
}

/// `sim *= iso`: `sim = sim * iso`. Upstream: `MulAssign<Isometry> for Similarity`.
pub impl SimilarityMatrix3MulAssignIsometryMatrix3<
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
> of MulAssign<SimilarityMatrix3<T>, IsometryMatrix3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: SimilarityMatrix3<T>, rhs: IsometryMatrix3<T>) {
        self = self.mul_isometry(rhs);
    }
}

/// `a *= b`: `a = a * b`. Upstream: `MulAssign<Similarity> for Similarity`.
pub impl SimilarityMatrix3MulAssign<
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
> of MulAssign<SimilarityMatrix3<T>, SimilarityMatrix3<T>> {
    #[inline(always)]
    fn mul_assign(ref self: SimilarityMatrix3<T>, rhs: SimilarityMatrix3<T>) {
        self = self * rhs;
    }
}

/// `sim /= iso`: `sim = sim * iso⁻¹`. Upstream: `DivAssign<Isometry> for Similarity`.
pub impl SimilarityMatrix3DivAssignIsometryMatrix3<
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
> of DivAssign<SimilarityMatrix3<T>, IsometryMatrix3<T>> {
    #[inline(always)]
    fn div_assign(ref self: SimilarityMatrix3<T>, rhs: IsometryMatrix3<T>) {
        self = self.div_isometry(rhs);
    }
}

/// `a /= b`: `a = a * b⁻¹`. Upstream: `DivAssign<Similarity> for Similarity`.
pub impl SimilarityMatrix3DivAssign<
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
> of DivAssign<SimilarityMatrix3<T>, SimilarityMatrix3<T>> {
    #[inline(always)]
    fn div_assign(ref self: SimilarityMatrix3<T>, rhs: SimilarityMatrix3<T>) {
        self = self * rhs.inverse();
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Similarity`.
pub impl SimilarityMatrix3Default<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Default<SimilarityMatrix3<T>> {
    #[inline(always)]
    fn default() -> SimilarityMatrix3<T> {
        let (o, l) = (R::zero(), R::one());
        SimilarityMatrix3 {
            isometry: IsometryMatrix3 {
                rotation: Rotation3 {
                    matrix: Matrix3 {
                        m11: l, m21: o, m31: o, m12: o, m22: l, m32: o, m13: o, m23: o, m33: l,
                    },
                },
                translation: Translation3 { vector: Vector3 { x: o, y: o, z: o } },
            },
            scaling: l,
        }
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Similarity`.
pub impl SimilarityMatrix3One<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<SimilarityMatrix3<T>> {
    #[inline(always)]
    fn one() -> SimilarityMatrix3<T> {
        SimilarityMatrix3Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @SimilarityMatrix3<T>) -> bool {
        *self == SimilarityMatrix3Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @SimilarityMatrix3<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `sim.into()`: the same similarity with its rotation as a unit quaternion (Shepperd's method).
/// Upstream: `SubsetOf<Similarity3> for SimilarityMatrix3` (`nalgebra::convert`).
pub impl Similarity3FromSimilarityMatrix3<
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
> of Into<SimilarityMatrix3<T>, Similarity3<T>> {
    #[inline(always)]
    fn into(self: SimilarityMatrix3<T>) -> Similarity3<T> {
        Similarity3 { isometry: self.isometry.into(), scaling: self.scaling }
    }
}

/// `sim.into()`: the same similarity with its rotation as a matrix (upstream's
/// quaternion-to-matrix form). Upstream: `SubsetOf<SimilarityMatrix3> for Similarity3`
/// (`nalgebra::convert`).
pub impl SimilarityMatrix3FromSimilarity3<
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
> of Into<Similarity3<T>, SimilarityMatrix3<T>> {
    #[inline(always)]
    fn into(self: Similarity3<T>) -> SimilarityMatrix3<T> {
        SimilarityMatrix3 { isometry: self.isometry.into(), scaling: self.scaling }
    }
}
