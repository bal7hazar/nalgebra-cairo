//! `Isometry2`: a 2D rigid-body transform, a rotation followed by a translation (upstream
//! `nalgebra::Isometry2`, which is `Isometry<T, UnitComplex<T>, 2>`).
//!
//! `iso * p = rotation * p + translation`. This is THE pose type of a physics engine: every body
//! position, collider placement and contact frame is an isometry (docs/research/01, §3).
//!
//! - `Isometry2Trait` / `Isometry2Impl`: construction from parts, composition, inverse, `inv_mul`,
//!   transforms, the in-place `append_*_mut`, the operator forms `mul_translation` /
//!   `mul_unit_complex` (Cairo's `Mul` is homogeneous) and `to_homogeneous` — everything that is
//!   algebraic, hence available for any `simba::scalar::Real` scalar (the fused kernels, the
//!   renormalisation of the rotation part and the trigonometry-free interpolation are
//!   crate-internal, WP 8.0);
//! - `Isometry2AngleTrait` / `Isometry2AngleImpl`: the constructors and the interpolation that go
//!   through an ANGLE (`new`, `rotation`, `lerp_slerp`), which additionally need
//!   `simba::scalar::Transcendental`;
//! - `a * b` (composition), `a / b`, `*=` / `/=`, `Default`, `One` and the conversions from a
//!   `Translation2`, a vector, a point or an array (and into a `Similarity2`): their impls live in
//!   this module, where the compiler finds them without any import. The rotation-MATRIX instance
//!   of upstream's generic `Isometry` is `IsometryMatrix2` (`.into()` converts between the two).
//!
//! The rotation is a `UnitComplex`, never a `Rotation2`: the two hold the same information, but the
//! complex form composes for 4 000 gas against 10 260 for the matrix and transforms a vector for
//! exactly the same price (see the module documentation of `rotation2`). Use
//! `to_homogeneous` when a matrix is what the consumer wants.
//!
//! Accuracy: the translation is carried EXACTLY through every operation whose rotation is the
//! identity, and the rotated parts inherit the 1-ulp floor of the `UnitComplex` kernels. Every
//! "rotate then translate" goes through the fused `rotate_translate` kernel (one rounding per
//! component). It gives the same bits as rotating and adding afterwards — `floor(x + t) =
//! floor(x) + t` for an integral `t` in raw units — for 23 % less gas, since the addition of a
//! `Fixed` pays an overflow check the accumulator does not
//! (`bench_isometry2_transform_point__alt_rotate_then_add`,
//! `test_transform_point_fused_and_composed_agree_bit_for_bit`).
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::num::traits::One;
use core::ops::{DivAssign, MulAssign};
use nalgebra_core::base::unit::Unit;
use nalgebra_types2::base::point2::Point2;
use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types2::geometry::rotation2::Rotation2;
use nalgebra_types2::geometry::translation2::Translation2;
use nalgebra_types3::base::matrix3::Matrix3;
use simba::scalar::{Real, Transcendental};
use crate::geometry::similarity2::{Similarity2, Similarity2Trait};
use crate::geometry::translation2::Translation2Trait;
use crate::geometry::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
use crate::internal::geometry::isometry2::Isometry2InternalTrait;

/// A 2D direct isometry: the rotation `rotation` followed by the translation `translation`. The
/// field names and their order are upstream's (`Isometry { rotation, translation }`); the parts
/// are read through the public fields, like upstream (the trait methods `translation` and
/// `rotation` are upstream's CONSTRUCTORS, not accessors).
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Isometry2<T> {
    pub rotation: UnitComplex<T>,
    pub translation: Translation2<T>,
}

/// Operations of `Isometry2<T>` that need no trigonometry, over a `Real` scalar. By value,
/// unrolled, no loop.
#[generate_trait]
pub impl Isometry2Impl<
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
> of Isometry2Trait<T> {
    // --- construction and parts ---------------------------------------------------------------

    /// The identity isometry (no rotation, no translation). Exact. Upstream:
    /// `Isometry2::identity`.
    #[inline(always)]
    fn identity() -> Isometry2<T> {
        Isometry2 {
            rotation: UnitComplex { re: R::one(), im: R::zero() },
            translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
        }
    }

    /// The isometry `translation ∘ rotation` (rotate first, then translate). Exact (two copies).
    /// Upstream: `Isometry2::from_parts`.
    #[inline(always)]
    fn from_parts(translation: Translation2<T>, rotation: UnitComplex<T>) -> Isometry2<T> {
        Isometry2 { rotation, translation }
    }

    /// The pure translation by `(x, y)`. Exact. Upstream: `Isometry2::translation`.
    #[inline(always)]
    fn translation(x: T, y: T) -> Isometry2<T> {
        Isometry2 {
            rotation: UnitComplex { re: R::one(), im: R::zero() },
            translation: Translation2 { vector: Vector2 { x, y } },
        }
    }

    // --- inverse and composition --------------------------------------------------------------

    /// The inverse isometry: the rotation is conjugated (exact) and the translation becomes
    /// `rotation⁻¹ · (-translation)`, i.e. two fused kernels on the negated vector —
    /// upstream's order, which matters in fixed point since `floor(-x) ≠ -floor(x)`. The
    /// negation is folded into the exact products (WP 11-OPT-1: the same bits, without its two
    /// `Fixed` negations, `test_inverse_matches_reference`), so a translation component equal to
    /// the scalar's `MIN` no longer panics on it. Panics on overflow of a result component and
    /// on `-MIN` of the rotation's imaginary part. Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: Isometry2<T>) -> Isometry2<T> {
        // `rotation.inverse_transform_vector(-translation)` with the negation folded into the
        // exact products: `re·(-x) + im·(-y)` is accumulated as `-re·x - im·y`, and
        // `re·(-y) - im·(-x)` is `im·x - re·y`, the same exact sums, so the same floors.
        let v = self.translation.vector;
        let r = self.rotation;
        Isometry2 {
            rotation: UnitComplex { re: r.re, im: -r.im },
            translation: Translation2 {
                vector: Vector2 {
                    x: R::wide_rescale(
                        R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), r.re, v.x), r.im, v.y),
                    ),
                    y: R::diff_prod(r.im, v.x, r.re, v.y),
                },
            },
        }
    }

    /// `self = self.inverse()` in place: the by-value form is the cheapest (a rotation and a
    /// translation to rebuild, nothing to reuse), so the bits are those of `inverse`. Panics as
    /// `inverse` does. Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: Isometry2<T>) {
        self = Self::inverse(self);
    }

    /// `self⁻¹ * other`, the RELATIVE pose rapier computes for every contact and joint
    /// (docs/research/01, §3.3), WITHOUT materialising the inverse: the translation is
    /// `rotation⁻¹ · (other.translation - self.translation)` (one exact subtraction and two
    /// fused kernels) and the rotation is `rotation⁻¹ · other.rotation` (two fused kernels with
    /// the conjugation folded in, `UnitComplex::rotation_to`).
    ///
    /// Measured 11 740 gas against 15 360 for `self.inverse() * other`, 1.31x, because the inverse
    /// rotates its own translation for nothing before the composition rotates it back
    /// (`bench_isometry2_inv_mul__alt_inverse_then_mul`). The two agree to 2 ulp but NOT bit for
    /// bit: the loser rounds the intermediate `rotation⁻¹ · (-translation)`
    /// (`test_inv_mul_alt_inverse_then_mul_differs_by_rounding`). Upstream: `inv_mul`.
    #[inline(always)]
    fn inv_mul(self: Isometry2<T>, other: Isometry2<T>) -> Isometry2<T> {
        let d = Vector2 {
            x: other.translation.vector.x - self.translation.vector.x,
            y: other.translation.vector.y - self.translation.vector.y,
        };
        Isometry2 {
            rotation: self.rotation.rotation_to(other.rotation),
            translation: Translation2 { vector: self.rotation.inverse_transform_vector(d) },
        }
    }

    // --- transforms ----------------------------------------------------------------------------

    /// `self * p = rotation · p + translation`, through `rotate_translate`: one rounding per
    /// component. Panics on overflow. Upstream: `transform_point` (`iso * p`).
    #[inline(always)]
    fn transform_point(self: Isometry2<T>, p: Point2<T>) -> Point2<T> {
        let c = Isometry2InternalTrait::rotate_translate(
            self.rotation, Vector2 { x: p.x, y: p.y }, self.translation.vector,
        );
        Point2 { x: c.x, y: c.y }
    }

    /// `rotation · v`: an isometry acts on a DISPLACEMENT by its rotation only (the translation
    /// cancels in `(p + v) - p`). Two fused kernels. Upstream: `transform_vector` (`iso * v`).
    #[inline(always)]
    fn transform_vector(self: Isometry2<T>, v: Vector2<T>) -> Vector2<T> {
        self.rotation.transform_vector(v)
    }

    /// `self⁻¹ * p = rotation⁻¹ · (p - translation)`: one exact subtraction and two fused
    /// kernels on the conjugate — no inverse isometry, no inverse rotation is ever built.
    /// Upstream:
    /// `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: Isometry2<T>, p: Point2<T>) -> Point2<T> {
        let d = Point2 { x: p.x - self.translation.vector.x, y: p.y - self.translation.vector.y };
        self.rotation.inverse_transform_point(d)
    }

    /// `rotation⁻¹ · v`: the inverse acting on a displacement, the conjugate applied directly
    /// (two fused kernels). Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: Isometry2<T>, v: Vector2<T>) -> Vector2<T> {
        self.rotation.inverse_transform_vector(v)
    }

    // --- append (in place) and the operator forms ----------------------------------------------

    /// `Translation(t) ∘ self`: the same rotation, the translation shifted by `t` (an exact
    /// addition), in place. Upstream: `append_translation_mut`.
    #[inline(always)]
    fn append_translation_mut(ref self: Isometry2<T>, t: Translation2<T>) {
        self =
            Isometry2 {
                rotation: self.rotation,
                translation: Translation2 {
                    vector: Vector2 {
                        x: self.translation.vector.x + t.vector.x,
                        y: self.translation.vector.y + t.vector.y,
                    },
                },
            };
    }

    /// `self * t = self ∘ Translation(t)`: the same rotation, the translation shifted by
    /// `rotation · t` (one `rotate_translate`). Upstream: `Mul<Translation2> for Isometry2` — a
    /// named method because Cairo's `Mul` is homogeneous (a documented rename,
    /// `scripts/api_parity.py`).
    #[inline(always)]
    fn mul_translation(self: Isometry2<T>, t: Translation2<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: self.rotation,
            translation: Translation2 {
                vector: Isometry2InternalTrait::rotate_translate(
                    self.rotation, t.vector, self.translation.vector,
                ),
            },
        }
    }

    /// `Rotation(r) ∘ self`: a rotation about the ORIGIN applied after `self`, so the translation
    /// is rotated too (`r · translation`) and the rotation becomes `r · rotation`, in place.
    /// Upstream:
    /// `append_rotation_mut`.
    #[inline(always)]
    fn append_rotation_mut(ref self: Isometry2<T>, r: UnitComplex<T>) {
        self =
            Isometry2 {
                rotation: r * self.rotation,
                translation: Translation2 { vector: r.transform_vector(self.translation.vector) },
            };
    }

    /// `self * r = self ∘ Rotation(r)`: a rotation applied BEFORE `self`, which leaves the
    /// translation untouched. Two fused kernels. Upstream: `Mul<UnitComplex> for Isometry2` — a
    /// named method because Cairo's `Mul` is homogeneous (a documented rename,
    /// `scripts/api_parity.py`).
    #[inline(always)]
    fn mul_unit_complex(self: Isometry2<T>, r: UnitComplex<T>) -> Isometry2<T> {
        Isometry2 { rotation: self.rotation * r, translation: self.translation }
    }

    /// The rotation `r` applied about the point `p` (which stays fixed): the translation becomes
    /// `r · (translation - p) + p`, the rotation `r · rotation`, in place. One exact subtraction,
    /// one `rotate_translate` and two fused kernels. Upstream: `append_rotation_wrt_point_mut`.
    fn append_rotation_wrt_point_mut(ref self: Isometry2<T>, r: UnitComplex<T>, p: Point2<T>) {
        let d = Vector2 { x: self.translation.vector.x - p.x, y: self.translation.vector.y - p.y };
        self =
            Isometry2 {
                rotation: r * self.rotation,
                translation: Translation2 {
                    vector: Isometry2InternalTrait::rotate_translate(
                        r, d, Vector2 { x: p.x, y: p.y },
                    ),
                },
            };
    }

    /// The rotation `r` applied about the isometry's own centre (the point `translation`): the
    /// translation is unchanged and the rotation becomes `r · rotation`, i.e. two fused kernels
    /// and nothing else (`append_rotation_wrt_point_mut` with `p = translation`, where the
    /// subtraction and the addition cancel exactly). Upstream: `append_rotation_wrt_center_mut`.
    #[inline(always)]
    fn append_rotation_wrt_center_mut(ref self: Isometry2<T>, r: UnitComplex<T>) {
        self = Isometry2 { rotation: r * self.rotation, translation: self.translation };
    }

    // --- conversions, renormalisation, comparison, interpolation -------------------------------

    /// The isometry as a 3x3 homogeneous matrix `[[re, -im, tx], [im, re, ty], [0, 0, 1]]`.
    /// Exact (copies and negations); panics on overflow (`-MIN`). Upstream: `to_homogeneous`.
    #[inline(always)]
    fn to_homogeneous(self: Isometry2<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.rotation.re,
            m21: self.rotation.im,
            m31: R::zero(),
            m12: -self.rotation.im,
            m22: self.rotation.re,
            m32: R::zero(),
            m13: self.translation.vector.x,
            m23: self.translation.vector.y,
            m33: R::one(),
        }
    }

    /// `true` when the two translations and the two rotations are within `ulps` smallest units
    /// (raw units for fixed point) of each other, component by component; cannot overflow. Note
    /// that `-rotation` is the same rotation and is NOT `abs_diff_eq` to it. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float
    /// epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Isometry2<T>, other: Isometry2<T>, ulps: u64) -> bool {
        self.translation.abs_diff_eq(other.translation, ulps)
            && self.rotation.abs_diff_eq(other.rotation, ulps)
    }

    // --- P09b completion ---------------------------------------------------------------------

    /// The rotation `r` about the point `p` (which stays fixed): translation `r · (-p) + p`
    /// through one `rotate_translate` (upstream: `r.transform_vector(-p) + p`, the same bits).
    /// Upstream: `Isometry::rotation_wrt_point`.
    #[inline(always)]
    fn rotation_wrt_point(r: UnitComplex<T>, p: Point2<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: r,
            translation: Translation2 {
                vector: Isometry2InternalTrait::rotate_translate(
                    r, Vector2 { x: -p.x, y: -p.y }, Vector2 { x: p.x, y: p.y },
                ),
            },
        }
    }

    /// `rotation · v` for a unit vector, re-wrapped WITHOUT renormalising (upstream's
    /// `Unit::new_unchecked`). Upstream: `Mul<Unit<Vector2>> for Isometry` (`iso * v`).
    #[inline(always)]
    fn transform_unit_vector(self: Isometry2<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>> {
        self.rotation.transform_unit_vector(v)
    }

    /// `rotation⁻¹ · v` for a unit vector, not renormalised. Upstream:
    /// `inverse_transform_unit_vector`.
    #[inline(always)]
    fn inverse_transform_unit_vector(self: Isometry2<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>> {
        self.rotation.inverse_transform_unit_vector(v)
    }

    /// `self / r = self * r⁻¹`: the rotation becomes `rotation / r` (two fused kernels on the
    /// conjugate), the translation is unchanged. Upstream: `Div<UnitComplex> for Isometry2` (a
    /// named method: Cairo's `Div` is homogeneous).
    #[inline(always)]
    fn div_unit_complex(self: Isometry2<T>, r: UnitComplex<T>) -> Isometry2<T> {
        Isometry2 { rotation: self.rotation / r, translation: self.translation }
    }

    /// `self * sim`: the similarity `(self * sim.isometry, sim.scaling)`. Upstream:
    /// `Mul<Similarity> for Isometry`.
    #[inline(always)]
    fn mul_similarity(self: Isometry2<T>, sim: Similarity2<T>) -> Similarity2<T> {
        Similarity2 { isometry: self * sim.isometry, scaling: sim.scaling }
    }

    /// `self / sim = self * sim⁻¹` (upstream's formula: the inverse similarity is materialised).
    /// Upstream: `Div<Similarity> for Isometry`.
    #[inline(always)]
    fn div_similarity(self: Isometry2<T>, sim: Similarity2<T>) -> Similarity2<T> {
        Self::mul_similarity(self, sim.inverse())
    }

    /// Alias of `to_homogeneous`. Upstream: `Isometry::to_matrix`.
    #[inline(always)]
    fn to_matrix(self: Isometry2<T>) -> Matrix3<T> {
        Self::to_homogeneous(self)
    }

    /// `relative_eq` of the translations and of the rotations (see
    /// `QuaternionTrait::relative_eq`). Upstream: `approx::RelativeEq::relative_eq` (DESIGN D3).
    fn relative_eq(self: Isometry2<T>, other: Isometry2<T>, epsilon: u64, max_relative: T) -> bool {
        self.translation.relative_eq(other.translation, epsilon, max_relative)
            && self.rotation.relative_eq(other.rotation, epsilon, max_relative)
    }

    /// `ulps_eq` of the translations and of the rotations (see `QuaternionTrait::ulps_eq`).
    /// Upstream: `approx::UlpsEq::ulps_eq` (DESIGN D3).
    fn ulps_eq(self: Isometry2<T>, other: Isometry2<T>, epsilon: u64, max_ulps: u32) -> bool {
        self.translation.ulps_eq(other.translation, epsilon, max_ulps)
            && self.rotation.ulps_eq(other.rotation, epsilon, max_ulps)
    }

    /// The same isometry with every scalar converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `Isometry2::cast` (and `SubsetOf<Isometry>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Isometry2<T>) -> Isometry2<U> {
        Isometry2 { rotation: self.rotation.cast(), translation: self.translation.cast() }
    }
}

/// Operations of `Isometry2<T>` that go through an angle, hence their own trait: scalars may
/// implement `Real` only (the whole rapier hot path — composition, `inv_mul`, transforms — is
/// in `Isometry2Trait` and needs no trigonometry at all).
///
/// Every method here costs at least one transcendental (`sin_cos` 16 800, `atan2` 15 400 gas on
/// `Fixed`), one to two orders of magnitude above the algebraic operations.
#[generate_trait]
pub impl Isometry2AngleImpl<
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
> of Isometry2AngleTrait<T> {
    /// The isometry that rotates by `angle` (radians) then translates by `translation`: one
    /// `sin_cos`. Upstream: `Isometry2::new(translation, angle)`.
    #[inline(always)]
    fn new(translation: Vector2<T>, angle: T) -> Isometry2<T> {
        let (sin, cos) = Tr::sin_cos(angle);
        Isometry2 {
            rotation: UnitComplex { re: cos, im: sin },
            translation: Translation2 { vector: translation },
        }
    }

    /// The pure rotation of `angle` radians about the origin: one `sin_cos`. Upstream:
    /// `Isometry2::rotation`.
    #[inline(always)]
    fn rotation(angle: T) -> Isometry2<T> {
        let (sin, cos) = Tr::sin_cos(angle);
        Isometry2 {
            rotation: UnitComplex { re: cos, im: sin },
            translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
        }
    }

    /// Interpolation between two poses: the translations linearly, the rotations spherically
    /// (`UnitComplex::slerp`, which takes the SHORTEST arc and walks it at constant angular
    /// velocity). `t` is not clamped; `t = 0` gives `self` exactly.
    ///
    /// 50 690 gas, dominated by one `atan2` and one `sin_cos`. The trigonometry-free
    /// normalized-lerp interpolation (crate-internal `lerp_nlerp`, benchmarked) gives the same path
    /// within a fraction of a degree for 18 520. Upstream:
    /// `Isometry2::lerp_slerp`. (Upstream has no `try_lerp_slerp` in 2D and neither does this
    /// port: `UnitComplex::slerp` is total, where `UnitQuaternion::try_slerp` can fail.)
    fn lerp_slerp(self: Isometry2<T>, other: Isometry2<T>, t: T) -> Isometry2<T> {
        Isometry2 {
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

/// `a * b`: the composition of two isometries, `b` applied first: the rotation is the product of
/// the rotations (two fused kernels) and the translation is `a.translation + a.rotation ·
/// b.translation` (one `rotate_translate`, so one rounding per component). Upstream: `Mul`.
pub impl Isometry2Mul<
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
> of Mul<Isometry2<T>> {
    #[inline(always)]
    fn mul(lhs: Isometry2<T>, rhs: Isometry2<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: lhs.rotation * rhs.rotation,
            translation: Translation2 {
                vector: Isometry2InternalTrait::rotate_translate(
                    lhs.rotation, rhs.translation.vector, lhs.translation.vector,
                ),
            },
        }
    }
}

/// `t.into()`: the pure translation as an isometry. Upstream: `From<Translation2> for Isometry2`.
pub impl Isometry2FromTranslation<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Translation2<T>, Isometry2<T>> {
    #[inline(always)]
    fn into(self: Translation2<T>) -> Isometry2<T> {
        Isometry2 { rotation: UnitComplex { re: R::one(), im: R::zero() }, translation: self }
    }
}

/// `a / b = a * b⁻¹`, upstream's formula (the inverse is materialised, then composed). Upstream:
/// `Div<Isometry> for Isometry`.
pub impl Isometry2Div<
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
> of Div<Isometry2<T>> {
    #[inline(always)]
    fn div(lhs: Isometry2<T>, rhs: Isometry2<T>) -> Isometry2<T> {
        lhs * rhs.inverse()
    }
}

/// `iso *= t`: `iso = iso * t` (`mul_translation`). Upstream: `MulAssign<Translation> for
/// Isometry`.
pub impl Isometry2MulAssignTranslation2<
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
> of MulAssign<Isometry2<T>, Translation2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Isometry2<T>, rhs: Translation2<T>) {
        self = self.mul_translation(rhs);
    }
}

/// `a *= b`: `a = a * b`. Upstream: `MulAssign<Isometry> for Isometry`.
pub impl Isometry2MulAssign<
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
> of MulAssign<Isometry2<T>, Isometry2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Isometry2<T>, rhs: Isometry2<T>) {
        self = self * rhs;
    }
}

/// `a /= b`: `a = a * b⁻¹`. Upstream: `DivAssign<Isometry> for Isometry`.
pub impl Isometry2DivAssign<
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
> of DivAssign<Isometry2<T>, Isometry2<T>> {
    #[inline(always)]
    fn div_assign(ref self: Isometry2<T>, rhs: Isometry2<T>) {
        self = self * rhs.inverse();
    }
}

/// `Default::default()`: the identity. Upstream: `Default for Isometry`.
pub impl Isometry2Default<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Default<Isometry2<T>> {
    #[inline(always)]
    fn default() -> Isometry2<T> {
        Isometry2 {
            rotation: UnitComplex { re: R::one(), im: R::zero() },
            translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
        }
    }
}

/// `One::one()`: the identity; `is_one` compares with it exactly. Upstream: `num::One for
/// Isometry`.
pub impl Isometry2One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<Isometry2<T>> {
    #[inline(always)]
    fn one() -> Isometry2<T> {
        Isometry2Default::<T>::default()
    }

    #[inline(always)]
    fn is_one(self: @Isometry2<T>) -> bool {
        *self == Isometry2Default::<T>::default()
    }

    #[inline(always)]
    fn is_non_one(self: @Isometry2<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `v.into()`: the pure translation by the vector `v`. Upstream: `From<SVector<T, 2>> for
/// Isometry`.
pub impl Isometry2FromVector2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Vector2<T>, Isometry2<T>> {
    #[inline(always)]
    fn into(self: Vector2<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: UnitComplex { re: R::one(), im: R::zero() },
            translation: Translation2 { vector: self },
        }
    }
}

/// `p.into()`: the pure translation by the coordinates of `p`. Upstream: `From<Point<T, 2>> for
/// Isometry`.
pub impl Isometry2FromPoint2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Point2<T>, Isometry2<T>> {
    #[inline(always)]
    fn into(self: Point2<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: UnitComplex { re: R::one(), im: R::zero() },
            translation: Translation2 { vector: Vector2 { x: self.x, y: self.y } },
        }
    }
}

/// `[x, y].into()`: the pure translation by `(x, y)`. Upstream: `From<[T; 2]> for Isometry`.
pub impl Isometry2FromArray<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Into<[T; 2], Isometry2<T>> {
    #[inline(always)]
    fn into(self: [T; 2]) -> Isometry2<T> {
        let [x, y] = self;
        Isometry2 {
            rotation: UnitComplex { re: R::one(), im: R::zero() },
            translation: Translation2 { vector: Vector2 { x, y } },
        }
    }
}

/// `iso.into()`: the similarity of scaling 1. Upstream: `SubsetOf<Similarity> for Isometry`
/// (`nalgebra::convert`).
pub impl Similarity2FromIsometry2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Isometry2<T>, Similarity2<T>> {
    #[inline(always)]
    fn into(self: Isometry2<T>) -> Similarity2<T> {
        Similarity2 { isometry: self, scaling: R::one() }
    }
}

/// `r.into()`: the isometry of rotation `r` (its first column as a unit complex) and zero
/// translation. Upstream: `SubsetOf<Isometry2> for Rotation2` (`nalgebra::convert(r)`).
pub impl Isometry2FromRotation2<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of Into<Rotation2<T>, Isometry2<T>> {
    #[inline(always)]
    fn into(self: Rotation2<T>) -> Isometry2<T> {
        Isometry2 {
            rotation: UnitComplex { re: self.matrix.m11, im: self.matrix.m21 },
            translation: Translation2 { vector: Vector2 { x: R::zero(), y: R::zero() } },
        }
    }
}


#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_types2::base::vector2::Vector2;
    use nalgebra_types2::geometry::translation2::Translation2;
    use crate::geometry::unit_complex::{UnitComplex, UnitComplexTrait};
    use super::{Isometry2, Isometry2Trait};

    /// `inverse` before WP 11-OPT-1: the translation is negated (two negations), then rotated by
    /// the conjugate (the new body folds the negation into the exact products).
    fn inverse_reference(self: Isometry2<Fixed>) -> Isometry2<Fixed> {
        let v = Vector2 { x: -self.translation.vector.x, y: -self.translation.vector.y };
        Isometry2 {
            rotation: UnitComplex { re: self.rotation.re, im: -self.rotation.im },
            translation: Translation2 { vector: self.rotation.inverse_transform_vector(v) },
        }
    }

    fn fx(raw: i64) -> Fixed {
        Fixed { raw }
    }

    fn iso(t: (i64, i64), r: (i64, i64)) -> Isometry2<Fixed> {
        let (x, y) = t;
        let (re, im) = r;
        Isometry2 {
            rotation: UnitComplex { re: fx(re), im: fx(im) },
            translation: Translation2 { vector: Vector2 { x: fx(x), y: fx(y) } },
        }
    }

    /// Deterministic 64-bit LCG (Knuth's MMIX constants).
    fn next(ref state: u128) -> u128 {
        state = (state * 6364136223846793005 + 1442695040888963407) % 0x10000000000000000;
        state
    }

    /// A raw value uniform in `[-bound, bound]`.
    fn draw(ref state: u128, bound: u128) -> i64 {
        let r: i128 = (next(ref state) % (2 * bound + 1)).try_into().unwrap();
        let b: i128 = bound.try_into().unwrap();
        (r - b).try_into().unwrap()
    }

    /// `inverse` against the reference, bit for bit: edge cases (identity, zero translation, a
    /// rotation by 180°, a near-180° rotation, translations near the scalar's range, the smallest
    /// raw values) and a deterministic sweep of normalised and unnormalised rotations with
    /// translations of every magnitude.
    #[test]
    fn test_inverse_matches_reference() {
        let one = 0x100000000;
        let big = 0x3fffffff00000000; // just under 2^30: |r| * |t| stays in range
        let rots = array![
            (one, 0), (-one, 0), (-one, 1), (0, one), (0, -one), (3037000500, 3037000500),
            (-3037000499, 3037000500), (1, -1), (0x80000000, -0x80000000),
        ];
        let trans = array![
            (0, 0), (one, -one), (6442450944, -9663676416), (1, -1), (big, -big), (-big, 0),
            (0x7fffffff, -0x80000000),
        ];
        let mut n = 0_u32;
        for r in rots.span() {
            for t in trans.span() {
                let x = iso(*t, *r);
                assert!(x.inverse() == inverse_reference(x));
                n += 1;
            }
        }
        let mut state: u128 = 0x15e2;
        for k in 0..240_u32 {
            let r = (draw(ref state, 0x100000000), draw(ref state, 0x100000000));
            let r = if k % 2 == 0 {
                let (re, im) = r;
                let c = UnitComplexTrait::new_normalize(Vector2 { x: fx(re), y: fx(im) });
                (c.re.raw, c.im.raw)
            } else {
                r
            };
            // Translation magnitudes from 2^-32 to 2^29 (|r| <= sqrt(2), so the products stay
            // in range).
            let tb: u128 = match k % 4 {
                0 => 0x100,
                1 => 0x100000000,
                2 => 0x100000000000,
                _ => 0x2000000000000000,
            };
            let x = iso((draw(ref state, tb), draw(ref state, tb)), r);
            assert!(x.inverse() == inverse_reference(x));
            n += 1;
        }
        assert!(n >= 200);
    }
}

// crate-map: generated items (tools/split/cratemap.py) [shapegen]
// crate-map: from base/matrix3.cairo
/// `isometry2.into()`: the homogeneous matrix. Exact (no arithmetic). Upstream: `From<Isometry2>
/// for Matrix3`.
pub impl Matrix3FromIsometry2<
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
> of Into<Isometry2<T>, Matrix3<T>> {
    #[inline(always)]
    fn into(self: Isometry2<T>) -> Matrix3<T> {
        Isometry2Trait::to_homogeneous(self)
    }
}
// crate-map: end
