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
//! - `a * b` (composition) and the conversion from a `Translation2`: their impls live in this
//!   module, where the compiler finds them without any import.
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

use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::Matrix3;
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;
use super::translation2::{Translation2, Translation2Trait};
use super::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

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
            rotation: UnitComplex { re: R::ONE, im: R::ZERO },
            translation: Translation2 { vector: Vector2 { x: R::ZERO, y: R::ZERO } },
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
            rotation: UnitComplex { re: R::ONE, im: R::ZERO },
            translation: Translation2 { vector: Vector2 { x, y } },
        }
    }

    // --- inverse and composition --------------------------------------------------------------

    /// The inverse isometry: the rotation is conjugated (exact) and the translation becomes
    /// `rotation⁻¹ · (-translation)`, i.e. two fused kernels on the negated vector —
    /// upstream's order, which matters in fixed point since `floor(-x) ≠ -floor(x)`. Panics on
    /// overflow (`-MIN` of a translation component). Upstream: `inverse`.
    #[inline(always)]
    fn inverse(self: Isometry2<T>) -> Isometry2<T> {
        let v = Vector2 { x: -self.translation.vector.x, y: -self.translation.vector.y };
        Isometry2 {
            rotation: UnitComplex { re: self.rotation.re, im: -self.rotation.im },
            translation: Translation2 { vector: self.rotation.inverse_transform_vector(v) },
        }
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
            m31: R::ZERO,
            m12: -self.rotation.im,
            m22: self.rotation.re,
            m32: R::ZERO,
            m13: self.translation.vector.x,
            m23: self.translation.vector.y,
            m33: R::ONE,
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
}

/// Crate-internal kernels of `Isometry2<T>` (WP 8.0: the public API is strictly upstream's): the
/// fused `rotate_translate` behind every "rotate then translate" (DESIGN D6), the renormalisation
/// of the rotation part (upstream renormalizes `iso.rotation` itself, in place) and the
/// trigonometry-free `lerp_nlerp` (upstream has `lerp_slerp` only).
#[generate_trait]
pub(crate) impl Isometry2InternalImpl<
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
> of Isometry2InternalTrait<T> {
    /// `r · v + t`, the kernel every "rotate then translate" of this type goes through: the two
    /// products AND the translation are accumulated exactly, then floored once per component.
    ///
    /// It gives the same bits as rotating and adding afterwards, since `floor(x + t) =
    /// floor(x) + t` for an integral `t` in raw units, for 4 400 gas instead of 5 680 (1.29x):
    /// a `Fixed` addition costs an overflow check (640 gas) that the accumulator does not pay.
    /// It also cannot overflow on the intermediate rotated vector, only on the result. Evidence:
    /// `bench_isometry2_transform_point__alt_rotate_then_add` and
    /// `test_transform_point_fused_and_composed_agree_bit_for_bit`.
    ///
    /// Not an upstream method: upstream writes `rotation * v + translation`, which in fixed point
    /// is exactly this kernel.
    #[inline(always)]
    fn rotate_translate(r: UnitComplex<T>, v: Vector2<T>, t: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: R::wide_rescale(
                R::wide_add(
                    R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), r.re, v.x), r.im, v.y), t.x,
                ),
            ),
            y: R::wide_rescale(
                R::wide_add(
                    R::wide_add_prod(R::wide_add_prod(R::wide_zero(), r.im, v.x), r.re, v.y), t.y,
                ),
            ),
        }
    }
    /// Renormalises the rotation exactly (`UnitComplex::renormalize`: one `norm2` and two exactly
    /// correctly rounded divisions), leaving the translation untouched. Call it after a long chain
    /// of compositions, each of which lets the norm of the complex drift by up to 2 ulp. Panics
    /// with `Fixed: division by zero` on a zero rotation. Upstream: `Rotation::renormalize` applied
    /// to the rotation part (upstream has no `Isometry::renormalize`).
    #[inline(always)]
    fn renormalize(self: Isometry2<T>) -> Isometry2<T> {
        let mut rotation = self.rotation;
        let _ = rotation.renormalize();
        Isometry2 { rotation, translation: self.translation }
    }
    /// Renormalises the rotation with one Newton step (`UnitComplex::renormalize_fast`: no square
    /// root, no division), for a norm already within about `2^-16` of 1. Saves 1.3 % over
    /// `renormalize` in Q32.32 — prefer `renormalize`, which converges from any norm. Upstream:
    /// `Unit::renormalize_fast` applied to the rotation part.
    #[inline(always)]
    fn renormalize_fast(self: Isometry2<T>) -> Isometry2<T> {
        let mut rotation = self.rotation;
        rotation.renormalize_fast();
        Isometry2 { rotation, translation: self.translation }
    }
    /// Interpolation WITHOUT trigonometry: the translations are interpolated linearly and the
    /// rotations by a normalised linear interpolation of the `(re, im)` pairs (the 2D counterpart
    /// of `UnitQuaternion::nlerp`, which `UnitComplex` does not provide). `t` is not clamped.
    ///
    /// Four fused `lerp`s, one `norm2` and two divisions: 18 520 gas against 50 690 for
    /// `lerp_slerp` (2.7x), which pays an `atan2` and a `sin_cos`
    /// (`bench_isometry2_lerp_slerp__*`). The angular velocity is NOT constant along the path (the
    /// chord is walked at constant speed, not the arc): the angle is off by at most
    /// `θ/2 - atan(tan(θ/2)·(2t-1))`-ish, i.e. below 2 % of the arc for a half turn and nothing
    /// for small angles. It takes the SHORTEST arc only when the two rotations are within a half
    /// turn; exactly opposite rotations make the interpolated pair vanish at `t = 1/2` and panic
    /// with `Fixed: division by zero`, like `UnitQuaternion::nlerp`.
    ///
    /// Upstream has no `Isometry2::lerp_nlerp`; this is `lerp_slerp` with `nlerp` in place of
    /// `slerp`, the form to use inside a physics step (DESIGN D6: transcendentals cost one to two
    /// orders of magnitude more than the algebra around them).
    fn lerp_nlerp(self: Isometry2<T>, other: Isometry2<T>, t: T) -> Isometry2<T> {
        let re = R::lerp(self.rotation.re, other.rotation.re, t);
        let im = R::lerp(self.rotation.im, other.rotation.im, t);
        let n = R::norm2(re, im);
        Isometry2 {
            rotation: UnitComplex { re: R::div(re, n), im: R::div(im, n) },
            translation: Translation2 {
                vector: Vector2 {
                    x: R::lerp(self.translation.vector.x, other.translation.vector.x, t),
                    y: R::lerp(self.translation.vector.y, other.translation.vector.y, t),
                },
            },
        }
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
            translation: Translation2 { vector: Vector2 { x: R::ZERO, y: R::ZERO } },
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
        Isometry2 { rotation: UnitComplex { re: R::ONE, im: R::ZERO }, translation: self }
    }
}
