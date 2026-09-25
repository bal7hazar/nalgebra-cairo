//! `UnitDualQuaternion`: a 3D rigid-body transform as a dual quaternion of unit norm (upstream
//! `nalgebra::UnitDualQuaternion`, WP 8.4-P12).
//!
//! `real` is the rotation (a unit quaternion) and `dual = t · real / 2` encodes the translation
//! `t`: the same transform as `Isometry3 { rotation: real, translation: t }` (`to_isometry` /
//! `from_isometry` convert between the two).
//!
//! - `UnitDualQuaternionTrait` / `UnitDualQuaternionImpl`: construction (`from_parts`,
//!   `from_isometry`, `from_rotation`), the `Unit` wrapper methods (`new_unchecked`,
//!   `new_normalize`, `try_new`, `into_inner`, `renormalize*`), conjugate / inverse, the parts
//!   (`rotation`, `translation`, `to_isometry`, `to_homogeneous`), the transforms, `lerp` /
//!   `nlerp`, the approximate comparisons and the products with `DualQuaternion`,
//!   `UnitQuaternion`, `Translation3` and `Isometry3` (`mul_<rhs>` / `div_<rhs>`: Cairo's
//!   `Mul` / `Div` are homogeneous) — everything algebraic, for any `simba::scalar::Real`;
//! - `UnitDualQuaternionAngleTrait` / `UnitDualQuaternionAngleImpl`: the screw-linear
//!   interpolation `sclerp` / `try_sclerp`, which additionally needs
//!   `simba::scalar::Transcendental`;
//! - `UnitQuaternionDualQuaternionTrait`, `Translation3DualQuaternionTrait`,
//!   `Isometry3DualQuaternionTrait`: upstream's `q * dq`, `t * dq`, `iso * dq` and their
//!   divisions, as methods of the left operand;
//! - `*`, `/`, unary `-`, `Default`, `One` and the conversions (`Isometry3` / `Matrix4` /
//!   `Similarity3` from a unit dual quaternion; a unit dual quaternion from an `Isometry3`, a
//!   `UnitQuaternion`, a `Rotation3` or a `Translation3`): their impls live in this module.
//!
//! Upstream this type is `Unit<DualQuaternion<T>>`; here it is a dedicated wrapper struct with the
//! same invariant (like `UnitQuaternion`), so that its operators live in this module. The
//! invariant `|real| = 1`, `real · dual* + dual · real* = 0` is a CONTRACT: `new_unchecked` does
//! not check it, and the dual quaternions produced here satisfy it within a few ulp, not exactly.
//! `(r, d)` and `(-r, -d)` are the same transform; no operation normalises the sign except
//! `sclerp` (shortest path), exactly like upstream.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently. The halving
//! of `from_parts` (`dual = t · real / 2`) and the doubling of `translation` (`t = 2 · dual ·
//! real*`) are folded into their accumulations (`Real::wide_mul_scalar`), so each output component
//! is floored ONCE for the whole expression.

use core::hash::{Hash, HashStateTrait};
use core::num::traits::One;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix4::Matrix4;
use crate::base::point3::Point3;
use crate::base::unit::Unit;
use crate::base::vector3::Vector3;
use super::dual_quaternion::{DualQuaternion, DualQuaternionTrait};
use super::isometry3::{Isometry3, Isometry3InternalTrait, Isometry3Trait};
use super::quaternion::{Quaternion, QuaternionInternalTrait, QuaternionTrait};
use super::rotation3::Rotation3;
use super::similarity3::{Similarity3, Similarity3Trait};
use super::translation3::Translation3;
use super::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};

/// Panic messages of `UnitDualQuaternion` (stable API).
pub mod errors {
    /// `sclerp` between two transforms whose rotations are orthogonal (`|real₁ · real₂|`
    /// within `default_epsilon` of zero): the interpolation path is undefined. Upstream panics with
    /// "DualQuaternion sclerp: ambiguous configuration.".
    pub const AMBIGUOUS_SCLERP: felt252 = 'nalgebra: ambiguous sclerp';
}

/// A 3D rigid-body transform: a dual quaternion of unit norm. Nothing enforces the invariant:
/// build it with `from_parts` / `from_isometry` / `from_rotation` / `new_normalize`, or with
/// `new_unchecked` when the dual quaternion is known to be a unit one. Upstream:
/// `UnitDualQuaternion<T>` (`= Unit<DualQuaternion<T>>`).
#[derive(Copy, Drop, PartialEq, Serde, Debug)]
pub struct UnitDualQuaternion<T> {
    pub dual_quaternion: DualQuaternion<T>,
}

/// Operations of `UnitDualQuaternion<T>` that need no trigonometry, over a `Real` scalar. By
/// value, unrolled, no loop.
#[generate_trait]
pub impl UnitDualQuaternionImpl<
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
> of UnitDualQuaternionTrait<T> {
    // --- construction ------------------------------------------------------------------------

    /// The identity transform `1 + ε·0`. Exact. Upstream: `UnitDualQuaternion::identity`.
    #[inline(always)]
    fn identity() -> UnitDualQuaternion<T> {
        UnitDualQuaternion { dual_quaternion: DualQuaternionTrait::identity() }
    }

    /// The transform `translation ∘ rotation` (rotate first, then translate): `real = rotation`,
    /// `dual = (0, translation) · rotation / 2`. Each component of the dual part is ONE
    /// accumulation of the three products of the pure Hamilton product, halved inside the
    /// accumulator (`Real::wide_mul_scalar` by `1/2`, exact): the exact floor of the true value,
    /// where upstream rounds the product then halves (exactly, in floating point). 12 products, 4
    /// roundings. Panics on overflow. Upstream: `UnitDualQuaternion::from_parts`.
    #[inline(always)]
    fn from_parts(
        translation: Translation3<T>, rotation: UnitQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        UnitDualQuaternion {
            dual_quaternion: DualQuaternion {
                real: rotation.quaternion,
                dual: UnitDualQuaternionInternalTrait::half_pure_mul(
                    translation.vector, rotation.quaternion,
                ),
            },
        }
    }

    /// `from_parts(isometry.translation, isometry.rotation)`. Upstream:
    /// `UnitDualQuaternion::from_isometry` (and `From<Isometry3>`).
    #[inline(always)]
    fn from_isometry(isometry: Isometry3<T>) -> UnitDualQuaternion<T> {
        Self::from_parts(isometry.translation, isometry.rotation)
    }

    /// The pure rotation `rotation + ε·0`. Exact. Upstream: `UnitDualQuaternion::from_rotation`.
    #[inline(always)]
    fn from_rotation(rotation: UnitQuaternion<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternion { dual_quaternion: DualQuaternionTrait::from_real(rotation.quaternion) }
    }

    /// Wraps `dq` WITHOUT normalizing it: the caller guarantees a unit dual quaternion. Upstream:
    /// `Unit::new_unchecked` on `Unit<DualQuaternion>`.
    #[inline(always)]
    fn new_unchecked(dq: DualQuaternion<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternion { dual_quaternion: dq }
    }

    /// `dq.normalize()` (both parts divided by the norm of the real part), wrapped. Panics with
    /// `Fixed: division by zero` on a zero real part. Upstream: `Unit::new_normalize` on
    /// `Unit<DualQuaternion>` (its `Normed` norm is the norm of the real part).
    #[inline(always)]
    fn new_normalize(dq: DualQuaternion<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternion { dual_quaternion: dq.normalize() }
    }

    /// `Some(new_normalize(dq))`, or `None` when the norm of the real part is `<= min_norm`. With
    /// `min_norm >= 0` it never divides by zero. Upstream: `Unit::try_new` on
    /// `Unit<DualQuaternion>`.
    #[inline(always)]
    fn try_new(dq: DualQuaternion<T>, min_norm: T) -> Option<UnitDualQuaternion<T>> {
        let n = dq.real.norm();
        if n <= min_norm {
            None
        } else {
            Some(
                UnitDualQuaternion {
                    dual_quaternion: DualQuaternion {
                        real: dq.real.unscale(n), dual: dq.dual.unscale(n),
                    },
                },
            )
        }
    }

    /// The underlying dual quaternion (a copy). Upstream: `Unit::into_inner`.
    #[inline(always)]
    fn into_inner(self: UnitDualQuaternion<T>) -> DualQuaternion<T> {
        self.dual_quaternion
    }

    /// The underlying dual quaternion (a copy: everything is by value here). Upstream:
    /// `UnitDualQuaternion::dual_quaternion` (`as_ref`).
    #[inline(always)]
    fn dual_quaternion(self: UnitDualQuaternion<T>) -> DualQuaternion<T> {
        self.dual_quaternion
    }

    /// The same transform with every scalar converted by `Into<T, U>` (the identity for the single
    /// scalar `Fixed`). Upstream: `UnitDualQuaternion::cast` (and
    /// `SubsetOf<UnitDualQuaternion>`).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: UnitDualQuaternion<T>) -> UnitDualQuaternion<U> {
        UnitDualQuaternion { dual_quaternion: self.dual_quaternion.cast() }
    }

    // --- renormalization ---------------------------------------------------------------------

    /// Renormalizes exactly, in place (`DualQuaternionTrait::normalize_mut`: both parts divided by
    /// the norm of the real part, one correctly rounded division per component), and returns the
    /// norm it had. Panics on a zero real part. Upstream: `Unit::renormalize` on
    /// `Unit<DualQuaternion>`.
    #[inline(always)]
    fn renormalize(ref self: UnitDualQuaternion<T>) -> T {
        let mut dq = self.dual_quaternion;
        let n = dq.normalize_mut();
        self = UnitDualQuaternion { dual_quaternion: dq };
        n
    }

    /// Renormalizes, in place, a dual quaternion whose real part has a norm already close to 1:
    /// one Newton step for the inverse square root, both parts times `(3 - |real|²) / 2` (the
    /// factor as ONE fused `mul_add`, as `UnitQuaternionTrait::renormalize_fast`), one product per
    /// component. No square root, no division. Upstream: `Unit::renormalize_fast` on
    /// `Unit<DualQuaternion>` (`scale_mut` of both parts).
    #[inline(always)]
    fn renormalize_fast(ref self: UnitDualQuaternion<T>) {
        let dq = self.dual_quaternion;
        let f = R::mul_add(dq.real.norm_squared(), -R::HALF, R::HALF + R::one());
        self = UnitDualQuaternion { dual_quaternion: dq.scale(f) };
    }

    // --- conjugate, inverse ------------------------------------------------------------------

    /// The dual-quaternion conjugate of both parts (`DualQuaternionTrait::conjugate`). Exact.
    /// Upstream: `UnitDualQuaternion::conjugate`.
    #[inline(always)]
    fn conjugate(self: UnitDualQuaternion<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternion { dual_quaternion: self.dual_quaternion.conjugate() }
    }

    /// `conjugate`, in place. Upstream: `conjugate_mut`.
    #[inline(always)]
    fn conjugate_mut(ref self: UnitDualQuaternion<T>) {
        self = UnitDualQuaternion { dual_quaternion: self.dual_quaternion.conjugate() };
    }

    /// The inverse transform: `real⁻¹ = real*` (the unit quaternion inverse, exact) and
    /// upstream's dual part `(-real*) · dual · real*`, in upstream's order: the negation is
    /// exact, then one fused Hamilton product and one fused product by the conjugate (`mul_conj`,
    /// the conjugate's signs folded in), each component floored once per product: 32 products, 8
    /// roundings.
    ///
    /// For an exactly unit dual quaternion this equals `dual*` (the quaternion conjugate of the
    /// dual part, free), but the rounded unit dual quaternions of fixed point are unit only within
    /// a few ulp, and `dual*` then differs from upstream's result by up to `|dual| · (|real|² -
    /// 1)`, i.e. thousands of ulp for translations of a few thousand units: the products are kept
    /// (fidelity to upstream, PLAN M8). Panics on overflow. Upstream:
    /// `UnitDualQuaternion::inverse`.
    #[inline(always)]
    fn inverse(self: UnitDualQuaternion<T>) -> UnitDualQuaternion<T> {
        let DualQuaternion { real, dual } = self.dual_quaternion;
        UnitDualQuaternion {
            dual_quaternion: DualQuaternion {
                real: real.conjugate(),
                dual: (Quaternion { i: real.i, j: real.j, k: real.k, w: -real.w } * dual)
                    .mul_conj(real),
            },
        }
    }

    /// `inverse`, in place. Upstream: `inverse_mut`.
    #[inline(always)]
    fn inverse_mut(ref self: UnitDualQuaternion<T>) {
        self = Self::inverse(self);
    }

    /// The transform `x` such that `x * self = other`, i.e. `other / self`. Upstream:
    /// `isometry_to`.
    #[inline(always)]
    fn isometry_to(
        self: UnitDualQuaternion<T>, other: UnitDualQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        other * Self::inverse(self)
    }

    // --- parts and conversions ---------------------------------------------------------------

    /// The rotation part, the real quaternion as a unit quaternion (a copy). Upstream: `rotation`.
    #[inline(always)]
    fn rotation(self: UnitDualQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternion { quaternion: self.dual_quaternion.real }
    }

    /// The translation part, `2 · (dual · real*).vector()`: each component ONE accumulation of
    /// the four products of the fused product by the conjugate, doubled inside the accumulator
    /// (`Real::wide_mul_scalar` by 2, exact) and floored once — the exact floor of the true
    /// value, where upstream rounds the product then doubles. 12 products, 3 roundings. Panics on
    /// overflow. Upstream: `translation`.
    #[inline(always)]
    fn translation(self: UnitDualQuaternion<T>) -> Translation3<T> {
        Translation3 {
            vector: UnitDualQuaternionInternalTrait::double_vector_mul_conj(
                self.dual_quaternion.dual, self.dual_quaternion.real,
            ),
        }
    }

    /// The isometry `Isometry3 { rotation: rotation(), translation: translation() }`. Upstream:
    /// `to_isometry` (and `From<UnitDualQuaternion> for Isometry3`).
    #[inline(always)]
    fn to_isometry(self: UnitDualQuaternion<T>) -> Isometry3<T> {
        Isometry3 { rotation: Self::rotation(self), translation: Self::translation(self) }
    }

    /// The 4x4 homogeneous matrix, `to_isometry().to_homogeneous()`. Upstream: `to_homogeneous`
    /// (and `From<UnitDualQuaternion> for Matrix4`).
    #[inline(always)]
    fn to_homogeneous(self: UnitDualQuaternion<T>) -> Matrix4<T> {
        Self::to_isometry(self).to_homogeneous()
    }

    // --- transforms --------------------------------------------------------------------------

    /// The transformed point `real · p · real* + translation` through the fused isometry kernel
    /// (`rotate_translate`) on `translation()`: 27 products, 6 roundings. Upstream's literal
    /// `((real · (0, p) + 2 · dual) · real*).vector()` is the same value in exact arithmetic;
    /// it costs 24 products but more gas, and rounds the scaled rotation `|real|² · p` instead of
    /// the unit sandwich (`bench_unit_dual_quaternion_transform_point__alt_upstream_sandwich`,
    /// both within the oracle tolerance). Panics on overflow. Upstream: `transform_point`
    /// (`dq * p`).
    #[inline(always)]
    fn transform_point(self: UnitDualQuaternion<T>, p: Point3<T>) -> Point3<T> {
        let t = Self::translation(self);
        let v = Isometry3InternalTrait::rotate_translate(
            Self::rotation(self), Vector3 { x: p.x, y: p.y, z: p.z }, t.vector,
        );
        Point3 { x: v.x, y: v.y, z: v.z }
    }

    /// The rotated vector (`UnitQuaternionTrait::transform_vector` of the real part; a vector is
    /// not translated). Upstream: `transform_vector` (`dq * v`).
    #[inline(always)]
    fn transform_vector(self: UnitDualQuaternion<T>, v: Vector3<T>) -> Vector3<T> {
        Self::rotation(self).transform_vector(v)
    }

    /// `transform_vector` of a unit vector, re-wrapped WITHOUT renormalising. Upstream:
    /// `Mul<Unit<Vector3>> for UnitDualQuaternion` (`dq * v`).
    #[inline(always)]
    fn transform_unit_vector(self: UnitDualQuaternion<T>, v: Unit<Vector3<T>>) -> Unit<Vector3<T>> {
        Unit { value: Self::rotation(self).transform_vector(v.value) }
    }

    /// `real* · (p - translation) · real`, WITHOUT forming the inverse dual quaternion: one
    /// `translation()`, three exact subtractions and one inverse rotation (27 products, 6
    /// roundings), where upstream's `self.inverse() * p` builds the inverse (32 products) and then
    /// transforms
    /// (`bench_unit_dual_quaternion_inverse_transform_point__alt_inverse_then_transform`, both
    /// within the oracle tolerance). Panics on overflow. Upstream: `inverse_transform_point`.
    #[inline(always)]
    fn inverse_transform_point(self: UnitDualQuaternion<T>, p: Point3<T>) -> Point3<T> {
        let t = Self::translation(self).vector;
        let v = Self::rotation(self)
            .inverse_transform_vector(Vector3 { x: p.x - t.x, y: p.y - t.y, z: p.z - t.z });
        Point3 { x: v.x, y: v.y, z: v.z }
    }

    /// The vector rotated by the inverse rotation (`UnitQuaternionTrait::inverse_transform_vector`
    /// of the real part, the same as upstream's `self.inverse() * v`, whose rotation is `real*`).
    /// Upstream: `inverse_transform_vector`.
    #[inline(always)]
    fn inverse_transform_vector(self: UnitDualQuaternion<T>, v: Vector3<T>) -> Vector3<T> {
        Self::rotation(self).inverse_transform_vector(v)
    }

    /// `inverse_transform_vector` of a unit vector, not renormalised. Upstream:
    /// `inverse_transform_unit_vector`.
    #[inline(always)]
    fn inverse_transform_unit_vector(
        self: UnitDualQuaternion<T>, v: Unit<Vector3<T>>,
    ) -> Unit<Vector3<T>> {
        Unit { value: Self::rotation(self).inverse_transform_vector(v.value) }
    }

    // --- interpolation -----------------------------------------------------------------------

    /// The linear interpolation of the two dual quaternions (`DualQuaternionTrait::lerp`), NOT a
    /// unit dual quaternion (see `nlerp`). Upstream: `UnitDualQuaternion::lerp`.
    #[inline(always)]
    fn lerp(self: UnitDualQuaternion<T>, other: UnitDualQuaternion<T>, t: T) -> DualQuaternion<T> {
        self.dual_quaternion.lerp(other.dual_quaternion, t)
    }

    /// Normalised linear interpolation: `lerp`, then `normalize` (both parts divided by the norm of
    /// the interpolated real part). Like upstream it does NOT take the shortest path (see
    /// `sclerp`). Panics with `Fixed: division by zero` when the interpolated real part is zero
    /// (opposite rotations at `t = 1/2`). Upstream: `nlerp`.
    #[inline(always)]
    fn nlerp(
        self: UnitDualQuaternion<T>, other: UnitDualQuaternion<T>, t: T,
    ) -> UnitDualQuaternion<T> {
        UnitDualQuaternion {
            dual_quaternion: self.dual_quaternion.lerp(other.dual_quaternion, t).normalize(),
        }
    }

    // --- approximate comparisons -------------------------------------------------------------

    /// `DualQuaternionTrait::abs_diff_eq` of the dual quaternions (`other` or `-other` as a whole,
    /// tolerance in ulp). Upstream: `approx::AbsDiffEq::abs_diff_eq`.
    #[inline(always)]
    fn abs_diff_eq(self: UnitDualQuaternion<T>, other: UnitDualQuaternion<T>, ulps: u64) -> bool {
        self.dual_quaternion.abs_diff_eq(other.dual_quaternion, ulps)
    }

    /// `DualQuaternionTrait::relative_eq` of the dual quaternions. Upstream:
    /// `approx::RelativeEq::relative_eq`.
    #[inline(always)]
    fn relative_eq(
        self: UnitDualQuaternion<T>, other: UnitDualQuaternion<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        self.dual_quaternion.relative_eq(other.dual_quaternion, epsilon, max_relative)
    }

    /// `DualQuaternionTrait::ulps_eq` of the dual quaternions. Upstream: `approx::UlpsEq::ulps_eq`.
    #[inline(always)]
    fn ulps_eq(
        self: UnitDualQuaternion<T>, other: UnitDualQuaternion<T>, epsilon: u64, max_ulps: u32,
    ) -> bool {
        self.dual_quaternion.ulps_eq(other.dual_quaternion, epsilon, max_ulps)
    }

    // --- heterogeneous operators (Cairo's operator traits are homogeneous) --------------------

    /// `self * rhs` with a general dual quaternion: the dual-quaternion product, a
    /// `DualQuaternion`. Upstream: `Mul<DualQuaternion> for UnitDualQuaternion`.
    #[inline(always)]
    fn mul_dual_quaternion(
        self: UnitDualQuaternion<T>, rhs: DualQuaternion<T>,
    ) -> DualQuaternion<T> {
        self.dual_quaternion * rhs
    }

    /// `self * r`: upstream's `self * from_rotation(r)`, whose zero dual part vanishes from the
    /// product: `(real · r, dual · r)`, two fused Hamilton products (bit-identical to the general
    /// fused product). Upstream: `Mul<UnitQuaternion> for UnitDualQuaternion`.
    #[inline(always)]
    fn mul_unit_quaternion(
        self: UnitDualQuaternion<T>, r: UnitQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        let DualQuaternion { real, dual } = self.dual_quaternion;
        UnitDualQuaternion {
            dual_quaternion: DualQuaternion {
                real: real * r.quaternion, dual: dual * r.quaternion,
            },
        }
    }

    /// `self / r = self * from_rotation(r⁻¹)`: `(real · r*, dual · r*)`, two fused products by
    /// the conjugate (`mul_conj`, bit-identical to multiplying by `r.inverse()`). Upstream:
    /// `Div<UnitQuaternion> for UnitDualQuaternion`.
    #[inline(always)]
    fn div_unit_quaternion(
        self: UnitDualQuaternion<T>, r: UnitQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        let DualQuaternion { real, dual } = self.dual_quaternion;
        UnitDualQuaternion {
            dual_quaternion: DualQuaternion {
                real: real.mul_conj(r.quaternion), dual: dual.mul_conj(r.quaternion),
            },
        }
    }

    /// `self * t`: upstream's `self * from_parts(t, identity)`, whose real part is `1`: `(real,
    /// dual + real · (0, t) / 2)`, the product by the pure quaternion halved inside its
    /// accumulation (floored once) then added exactly. Panics on overflow. Upstream:
    /// `Mul<Translation3> for UnitDualQuaternion`.
    #[inline(always)]
    fn mul_translation(self: UnitDualQuaternion<T>, t: Translation3<T>) -> UnitDualQuaternion<T> {
        let DualQuaternion { real, dual } = self.dual_quaternion;
        UnitDualQuaternion {
            dual_quaternion: DualQuaternion {
                real, dual: dual + UnitDualQuaternionInternalTrait::half_mul_pure(real, t.vector),
            },
        }
    }

    /// `self / t = self * t⁻¹`: `mul_translation` by the negated translation (exact negation, as
    /// upstream's `t.inverse()`). Upstream: `Div<Translation3> for UnitDualQuaternion`.
    #[inline(always)]
    fn div_translation(self: UnitDualQuaternion<T>, t: Translation3<T>) -> UnitDualQuaternion<T> {
        let v = t.vector;
        Self::mul_translation(self, Translation3 { vector: Vector3 { x: -v.x, y: -v.y, z: -v.z } })
    }

    /// `self * from_isometry(iso)`. Upstream: `Mul<Isometry3> for UnitDualQuaternion`.
    #[inline(always)]
    fn mul_isometry(self: UnitDualQuaternion<T>, iso: Isometry3<T>) -> UnitDualQuaternion<T> {
        self * Self::from_isometry(iso)
    }

    /// `self / from_isometry(iso)`. Upstream: `Div<Isometry3> for UnitDualQuaternion`.
    #[inline(always)]
    fn div_isometry(self: UnitDualQuaternion<T>, iso: Isometry3<T>) -> UnitDualQuaternion<T> {
        self * Self::inverse(Self::from_isometry(iso))
    }
}

/// Crate-internal kernels of `UnitDualQuaternion<T>`: the products by a pure quaternion `(0, t)`
/// with the halving folded into the accumulation (`from_parts`, the translation operators) and
/// the doubled imaginary part of `dual · real*` (`translation`).
#[generate_trait]
pub(crate) impl UnitDualQuaternionInternalImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of UnitDualQuaternionInternalTrait<T> {
    /// `((0, t) · r) / 2`, each component ONE accumulation of three products (the Hamilton product
    /// with a zero real part on the left), halved exactly and floored once.
    fn half_pure_mul(t: Vector3<T>, r: Quaternion<T>) -> Quaternion<T> {
        // w = -tx·ri - ty·rj - tz·rk
        let w = R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), t.x, r.i), t.y, r.j);
        let w = R::wide_mul_scalar(R::wide_sub_prod(w, t.z, r.k), R::HALF);
        // i = tx·rw + ty·rk - tz·rj
        let i = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), t.x, r.w), t.y, r.k);
        let i = R::wide_mul_scalar(R::wide_sub_prod(i, t.z, r.j), R::HALF);
        // j = -tx·rk + ty·rw + tz·ri
        let j = R::wide_add_prod(R::wide_sub_prod(R::wide_zero(), t.x, r.k), t.y, r.w);
        let j = R::wide_mul_scalar(R::wide_add_prod(j, t.z, r.i), R::HALF);
        // k = tx·rj - ty·ri + tz·rw
        let k = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), t.x, r.j), t.y, r.i);
        let k = R::wide_mul_scalar(R::wide_add_prod(k, t.z, r.w), R::HALF);
        Quaternion { i, j, k, w }
    }

    /// `(r · (0, t)) / 2`, each component ONE accumulation of three products (the Hamilton product
    /// with a zero real part on the right), halved exactly and floored once.
    fn half_mul_pure(r: Quaternion<T>, t: Vector3<T>) -> Quaternion<T> {
        // w = -ri·tx - rj·ty - rk·tz
        let w = R::wide_sub_prod(R::wide_sub_prod(R::wide_zero(), r.i, t.x), r.j, t.y);
        let w = R::wide_mul_scalar(R::wide_sub_prod(w, r.k, t.z), R::HALF);
        // i = rw·tx + rj·tz - rk·ty
        let i = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), r.w, t.x), r.j, t.z);
        let i = R::wide_mul_scalar(R::wide_sub_prod(i, r.k, t.y), R::HALF);
        // j = rw·ty - ri·tz + rk·tx
        let j = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), r.w, t.y), r.i, t.z);
        let j = R::wide_mul_scalar(R::wide_add_prod(j, r.k, t.x), R::HALF);
        // k = rw·tz + ri·ty - rj·tx
        let k = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), r.w, t.z), r.i, t.y);
        let k = R::wide_mul_scalar(R::wide_sub_prod(k, r.j, t.x), R::HALF);
        Quaternion { i, j, k, w }
    }

    /// `2 · (d · r*).vector()`, each component ONE accumulation of the four products of
    /// `QuaternionInternalTrait::mul_conj`'s imaginary part, doubled exactly and floored once.
    fn double_vector_mul_conj(d: Quaternion<T>, r: Quaternion<T>) -> Vector3<T> {
        // i = -dw·ri + di·rw - dj·rk + dk·rj
        let x = R::wide_sub_prod(R::wide_zero(), d.w, r.i);
        let x = R::wide_sub_prod(R::wide_add_prod(x, d.i, r.w), d.j, r.k);
        let x = R::wide_mul_scalar(R::wide_add_prod(x, d.k, r.j), R::TWO);
        // j = -dw·rj + di·rk + dj·rw - dk·ri
        let y = R::wide_sub_prod(R::wide_zero(), d.w, r.j);
        let y = R::wide_add_prod(R::wide_add_prod(y, d.i, r.k), d.j, r.w);
        let y = R::wide_mul_scalar(R::wide_sub_prod(y, d.k, r.i), R::TWO);
        // k = -dw·rk - di·rj + dj·ri + dk·rw
        let z = R::wide_sub_prod(R::wide_zero(), d.w, r.k);
        let z = R::wide_add_prod(R::wide_sub_prod(z, d.i, r.j), d.j, r.i);
        let z = R::wide_mul_scalar(R::wide_add_prod(z, d.k, r.w), R::TWO);
        Vector3 { x, y, z }
    }
}

/// The screw-linear interpolation of `UnitDualQuaternion<T>`, over a `Real` + `Transcendental`
/// scalar (the split mirrors `UnitQuaternionTrait` / `UnitQuaternionAngleTrait`).
#[generate_trait]
pub impl UnitDualQuaternionAngleImpl<
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
> of UnitDualQuaternionAngleTrait<T> {
    /// `try_sclerp` with `epsilon = default_epsilon` (1 ulp), panicking with
    /// `errors::AMBIGUOUS_SCLERP` where it returns `None`. Upstream: `sclerp` (which passes
    /// `T::default_epsilon()` and panics likewise).
    fn sclerp(
        self: UnitDualQuaternion<T>, other: UnitDualQuaternion<T>, t: T,
    ) -> UnitDualQuaternion<T> {
        Self::try_sclerp(self, other, t, R::default_epsilon()).expect(errors::AMBIGUOUS_SCLERP)
    }

    /// The screw-linear interpolation from `self` (`t = 0`) to `other` (`t = 1`): the constant
    /// screw motion (rotation about an axis together with a translation along it) between the two
    /// transforms, taking the shortest path (`other` is negated when the real parts have a negative
    /// dot product). Upstream's algorithm, step by step:
    ///
    /// - `None` when `|real₁ · real₂| <= epsilon` (orthogonal rotations: the direction of the
    ///   path is undefined; upstream's `relative_eq!(dot, 0, epsilon)`);
    /// - `diff = self.conjugate() * other`; when `|diff.real.vector()|² <= epsilon` (floored, the
    ///   rotations are equal to about `sqrt(epsilon)`), the translations are linearly
    ///   interpolated and the rotation of `self` is kept, like upstream;
    /// - otherwise the screw parameters (half angle `acos(diff.real.w)`, pitch, direction,
    ///   moment) are scaled by `t` and the resulting screw is applied after `self`.
    ///
    /// Fixed-point formulation: `1 / |v|` (with `v = diff.real.vector()`) is not formed as
    /// upstream's `sqrt(1 / |v|²)` — which overflows for the smallest accepted `|v|` — but
    /// each quantity divides by `|v| = Real::norm3(v)` once (correctly rounded); the argument of
    /// `acos`
    /// is clamped to `[-1, 1]` (a rounded unit quaternion may exceed it by an ulp, where upstream's
    /// `acos` would return `NaN`); the half angle is `acos(w) · t` directly (upstream doubles it,
    /// scales it and halves it again); the products of three factors are fused. `epsilon` is in
    /// scalar units (DESIGN D3). Upstream: `try_sclerp`.
    fn try_sclerp(
        self: UnitDualQuaternion<T>, other: UnitDualQuaternion<T>, t: T, epsilon: T,
    ) -> Option<UnitDualQuaternion<T>> {
        let dot = self.dual_quaternion.real.dot(other.dual_quaternion.real);
        if R::abs(dot) <= epsilon {
            return None;
        }
        let other = if R::is_sign_negative(dot) {
            -other
        } else {
            other
        };
        let diff = self.dual_quaternion.conjugate() * other.dual_quaternion;
        let (dr, dd) = (diff.real, diff.dual);
        if R::norm_squared3(dr.i, dr.j, dr.k) <= epsilon {
            let (a, b) = (self.translation().vector, other.translation().vector);
            return Some(
                UnitDualQuaternionTrait::from_parts(
                    Translation3 {
                        vector: Vector3 {
                            x: R::lerp(a.x, b.x, t),
                            y: R::lerp(a.y, b.y, t),
                            z: R::lerp(a.z, b.z, t),
                        },
                    },
                    self.rotation(),
                ),
            );
        }
        let nv = R::norm3(dr.i, dr.j, dr.k);
        // Screw parameters of `diff`: direction `v / |v|`, pitch `-2·dual.w / |v|`, moment
        // `(dual.vector - direction · pitch · real.w / 2) / |v|`.
        let (ux, uy, uz) = R::div3(dr.i, dr.j, dr.k, nv);
        let pitch = R::div(-(dd.w + dd.w), nv);
        let c = R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), pitch, dr.w), R::HALF);
        let (mx, my, mz) = R::div3(
            R::wide_rescale(R::wide_sub_prod(R::wide_add(R::wide_zero(), dd.i), ux, c)),
            R::wide_rescale(R::wide_sub_prod(R::wide_add(R::wide_zero(), dd.j), uy, c)),
            R::wide_rescale(R::wide_sub_prod(R::wide_add(R::wide_zero(), dd.k), uz, c)),
            nv,
        );
        // Scaled by `t`: half angle `acos(real.w) · t`, half pitch `pitch · t / 2`.
        let half_angle = Tr::acos(R::clamp(dr.w, -R::one(), R::one())) * t;
        let hp = R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), pitch, t), R::HALF);
        let (sin, cos) = Tr::sin_cos(half_angle);
        let hpc = hp * cos;
        let screw = DualQuaternion {
            real: Quaternion { i: ux * sin, j: uy * sin, k: uz * sin, w: cos },
            dual: Quaternion {
                i: R::sum_prod2(mx, sin, ux, hpc),
                j: R::sum_prod2(my, sin, uy, hpc),
                k: R::sum_prod2(mz, sin, uz, hpc),
                w: R::wide_rescale(R::wide_sub_prod(R::wide_zero(), hp, sin)),
            },
        };
        Some(UnitDualQuaternion { dual_quaternion: self.dual_quaternion * screw })
    }
}

/// `q * dq` and `q / dq` for a `UnitQuaternion` on the left (Cairo's `Mul` / `Div` are
/// homogeneous, so upstream's heterogeneous operators are methods of the left operand).
#[generate_trait]
pub impl UnitQuaternionDualQuaternionImpl<
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
> of UnitQuaternionDualQuaternionTrait<T> {
    /// `self * rhs`: upstream's `from_rotation(self) * rhs`, whose zero dual part vanishes from
    /// the product: `(self · real, self · dual)`, two fused Hamilton products (bit-identical to
    /// the general fused product). Upstream: `Mul<UnitDualQuaternion> for UnitQuaternion`.
    #[inline(always)]
    fn mul_unit_dual_quaternion(
        self: UnitQuaternion<T>, rhs: UnitDualQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        let DualQuaternion { real, dual } = rhs.dual_quaternion;
        UnitDualQuaternion {
            dual_quaternion: DualQuaternion {
                real: self.quaternion * real, dual: self.quaternion * dual,
            },
        }
    }

    /// `self / rhs = from_rotation(self) * rhs.inverse()` (`UnitDualQuaternionTrait::inverse`,
    /// then two fused Hamilton products). Upstream: `Div<UnitDualQuaternion> for UnitQuaternion`.
    #[inline(always)]
    fn div_unit_dual_quaternion(
        self: UnitQuaternion<T>, rhs: UnitDualQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        Self::mul_unit_dual_quaternion(self, rhs.inverse())
    }
}

/// `t * dq` and `t / dq` for a `Translation3` on the left (methods of the left operand, see
/// `UnitQuaternionDualQuaternionTrait`).
#[generate_trait]
pub impl Translation3DualQuaternionImpl<
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
> of Translation3DualQuaternionTrait<T> {
    /// `self * rhs`: upstream's `from_parts(self, identity) * rhs`, whose real part is `1`:
    /// `(real, dual + (0, t) · real / 2)`, the product by the pure quaternion halved inside its
    /// accumulation (floored once) then added exactly. Panics on overflow. Upstream:
    /// `Mul<UnitDualQuaternion> for Translation3`.
    #[inline(always)]
    fn mul_unit_dual_quaternion(
        self: Translation3<T>, rhs: UnitDualQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        let DualQuaternion { real, dual } = rhs.dual_quaternion;
        UnitDualQuaternion {
            dual_quaternion: DualQuaternion {
                real,
                dual: dual + UnitDualQuaternionInternalTrait::half_pure_mul(self.vector, real),
            },
        }
    }

    /// `self / rhs = from_parts(self, identity) * rhs.inverse()`: `mul_unit_dual_quaternion` by
    /// the inverse. Upstream: `Div<UnitDualQuaternion> for Translation3`.
    #[inline(always)]
    fn div_unit_dual_quaternion(
        self: Translation3<T>, rhs: UnitDualQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        Self::mul_unit_dual_quaternion(self, rhs.inverse())
    }
}

/// `iso * dq` and `iso / dq` for an `Isometry3` on the left (methods of the left operand, see
/// `UnitQuaternionDualQuaternionTrait`).
#[generate_trait]
pub impl Isometry3DualQuaternionImpl<
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
> of Isometry3DualQuaternionTrait<T> {
    /// `from_isometry(self) * rhs`. Upstream: `Mul<UnitDualQuaternion> for Isometry3`.
    #[inline(always)]
    fn mul_unit_dual_quaternion(
        self: Isometry3<T>, rhs: UnitDualQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        UnitDualQuaternionTrait::from_isometry(self) * rhs
    }

    /// `from_isometry(self) / rhs`. Upstream: `Div<UnitDualQuaternion> for Isometry3`.
    #[inline(always)]
    fn div_unit_dual_quaternion(
        self: Isometry3<T>, rhs: UnitDualQuaternion<T>,
    ) -> UnitDualQuaternion<T> {
        UnitDualQuaternionTrait::from_isometry(self) * rhs.inverse()
    }
}

/// `Hash` of the eight components, in the `[T; 8]` order of `DualQuaternion`'s `Serde`. Written by
/// hand because `DualQuaternion` itself is not `Hash` (upstream's `DualQuaternion` has no `Hash`,
/// and upstream's `Hash for Unit<T: Hash>` would not apply to it); it keeps `UnitDualQuaternion`
/// on par with the other `Unit` types of this library (`UnitQuaternion`, `UnitComplex`, `Unit`),
/// which are `Hash`. Upstream: `Hash for Unit`.
pub impl UnitDualQuaternionHash<
    T, S, +HashStateTrait<S>, +Drop<S>, +Hash<T, S>, +Copy<T>, +Drop<T>,
> of Hash<UnitDualQuaternion<T>, S> {
    fn update_state(state: S, value: UnitDualQuaternion<T>) -> S {
        let DualQuaternion { real, dual } = value.dual_quaternion;
        let state = Hash::update_state(state, real.i);
        let state = Hash::update_state(state, real.j);
        let state = Hash::update_state(state, real.k);
        let state = Hash::update_state(state, real.w);
        let state = Hash::update_state(state, dual.i);
        let state = Hash::update_state(state, dual.j);
        let state = Hash::update_state(state, dual.k);
        Hash::update_state(state, dual.w)
    }
}

/// `a * b`: the composition of the two transforms, `b` applied first (the fused
/// dual-quaternion product, `DualQuaternionMul`). Upstream: `Mul`.
pub impl UnitDualQuaternionMul<
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
> of Mul<UnitDualQuaternion<T>> {
    #[inline(always)]
    fn mul(lhs: UnitDualQuaternion<T>, rhs: UnitDualQuaternion<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternion { dual_quaternion: lhs.dual_quaternion * rhs.dual_quaternion }
    }
}

/// `a / b = a * b.inverse()` (`UnitDualQuaternionTrait::inverse`, then the fused product).
/// Upstream: `Div`.
pub impl UnitDualQuaternionDiv<
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
> of Div<UnitDualQuaternion<T>> {
    #[inline(always)]
    fn div(lhs: UnitDualQuaternion<T>, rhs: UnitDualQuaternion<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternion { dual_quaternion: lhs.dual_quaternion * rhs.inverse().dual_quaternion }
    }
}

/// `-a`: both parts negated, the same transform. Exact; panics on overflow (`-MIN`). Upstream:
/// `Neg`.
pub impl UnitDualQuaternionNeg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<UnitDualQuaternion<T>> {
    #[inline(always)]
    fn neg(a: UnitDualQuaternion<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternion { dual_quaternion: -a.dual_quaternion }
    }
}

/// `Default::default()`: the identity transform, like upstream (`Default` for
/// `UnitDualQuaternion` is `identity`).
pub impl UnitDualQuaternionDefault<T, impl R: Real<T>, +Drop<T>> of Default<UnitDualQuaternion<T>> {
    #[inline(always)]
    fn default() -> UnitDualQuaternion<T> {
        UnitDualQuaternion {
            dual_quaternion: DualQuaternion {
                real: Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::one() },
                dual: Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::zero() },
            },
        }
    }
}

/// `One::one()`: the identity transform; `is_one` compares the eight components with `1 + ε·0`
/// exactly. Upstream: `num::One for UnitDualQuaternion`.
pub impl UnitDualQuaternionOne<
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
> of One<UnitDualQuaternion<T>> {
    #[inline(always)]
    fn one() -> UnitDualQuaternion<T> {
        UnitDualQuaternionTrait::identity()
    }

    #[inline(always)]
    fn is_one(self: @UnitDualQuaternion<T>) -> bool {
        self.dual_quaternion.is_one()
    }

    #[inline(always)]
    fn is_non_one(self: @UnitDualQuaternion<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `dq.into()`: `to_isometry`. Upstream: `From<UnitDualQuaternion> for Isometry3` (and
/// `SubsetOf<Isometry3> for UnitDualQuaternion`, `nalgebra::convert`).
pub impl Isometry3FromUnitDualQuaternion<
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
> of Into<UnitDualQuaternion<T>, Isometry3<T>> {
    #[inline(always)]
    fn into(self: UnitDualQuaternion<T>) -> Isometry3<T> {
        self.to_isometry()
    }
}

/// `dq.into()`: `to_homogeneous`. Upstream: `From<UnitDualQuaternion> for Matrix4`.
pub impl Matrix4FromUnitDualQuaternion<
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
> of Into<UnitDualQuaternion<T>, Matrix4<T>> {
    #[inline(always)]
    fn into(self: UnitDualQuaternion<T>) -> Matrix4<T> {
        self.to_homogeneous()
    }
}

/// `dq.into()`: the similarity of isometry `to_isometry()` and scaling 1. Upstream:
/// `SubsetOf<Similarity3> for UnitDualQuaternion` (`nalgebra::convert`).
pub impl Similarity3FromUnitDualQuaternion<
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
> of Into<UnitDualQuaternion<T>, Similarity3<T>> {
    #[inline(always)]
    fn into(self: UnitDualQuaternion<T>) -> Similarity3<T> {
        Similarity3Trait::from_isometry(self.to_isometry(), R::one())
    }
}

/// `iso.into()`: `from_isometry`. Upstream: `From<Isometry3> for UnitDualQuaternion` (and
/// `SubsetOf<UnitDualQuaternion> for Isometry3`).
pub impl UnitDualQuaternionFromIsometry3<
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
> of Into<Isometry3<T>, UnitDualQuaternion<T>> {
    #[inline(always)]
    fn into(self: Isometry3<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternionTrait::from_isometry(self)
    }
}

/// `q.into()`: `from_rotation`. Upstream: `SubsetOf<UnitDualQuaternion> for UnitQuaternion`
/// (`nalgebra::convert`).
pub impl UnitDualQuaternionFromUnitQuaternion<
    T, impl R: Real<T>, +Drop<T>,
> of Into<UnitQuaternion<T>, UnitDualQuaternion<T>> {
    #[inline(always)]
    fn into(self: UnitQuaternion<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternion {
            dual_quaternion: DualQuaternion {
                real: self.quaternion,
                dual: Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::zero() },
            },
        }
    }
}

/// `r.into()`: `from_rotation(UnitQuaternion::from_rotation_matrix(r))`. Upstream:
/// `SubsetOf<UnitDualQuaternion> for Rotation3` (`nalgebra::convert`).
pub impl UnitDualQuaternionFromRotation3<
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
> of Into<Rotation3<T>, UnitDualQuaternion<T>> {
    #[inline(always)]
    fn into(self: Rotation3<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternionTrait::from_rotation(UnitQuaternionTrait::from_rotation_matrix(self))
    }
}

/// `t.into()`: `from_parts(t, identity)`, i.e. `(1, (0, t / 2))` (the halving floored once).
/// Upstream: `SubsetOf<UnitDualQuaternion> for Translation3` (`nalgebra::convert`).
pub impl UnitDualQuaternionFromTranslation3<
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
> of Into<Translation3<T>, UnitDualQuaternion<T>> {
    #[inline(always)]
    fn into(self: Translation3<T>) -> UnitDualQuaternion<T> {
        UnitDualQuaternionTrait::from_parts(self, UnitQuaternionTrait::identity())
    }
}
