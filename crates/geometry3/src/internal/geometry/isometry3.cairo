//! Internal, no stability promise: the crate-private items of `geometry::isometry3` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_static3::base::vector3::Vector3Trait;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::geometry::translation3::Translation3;
use simba::scalar::Real;
use crate::geometry::isometry3::Isometry3;
use crate::geometry::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};

/// Crate-internal kernels of `Isometry3<T>` (WP 8.0: the public API is strictly upstream's): the
/// fused `rotate_translate` behind every "rotate then translate" (DESIGN D6), the renormalisation
/// of the rotation part (upstream renormalizes `iso.rotation` itself, in place) and the
/// trigonometry-free `lerp_nlerp` (upstream has `lerp_slerp` only).
#[generate_trait]
pub impl Isometry3InternalImpl<
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
> of Isometry3InternalTrait<T> {
    /// `r · v + t`, the kernel every "rotate then translate" of this type goes through: the
    /// quaternion sandwich `v + 2w·(u × v) + 2u × (u × v)` of
    /// `UnitQuaternion::transform_vector` with the translation folded into its wide accumulation,
    /// so each output component is floored ONCE for the whole expression (15 products, 3
    /// roundings, no division).
    ///
    /// It gives the same bits as rotating and adding afterwards, since `floor(x + t) =
    /// floor(x) + t` for an integral `t` in raw units, for 24 430 gas instead of 25 750 (5 %
    /// cheaper): a `Fixed` addition costs an overflow check (540 gas) that the accumulator does
    /// not pay. It also cannot overflow on the intermediate rotated vector, only on the result.
    /// Duplicating the sandwich (instead of calling `UnitQuaternion::transform_vector` and
    /// adding) is the price of that single rounding; it is written once here and reused by
    /// `transform_point`, `Mul`, `mul_translation` and `append_rotation_wrt_point_mut`.
    /// Evidence: `bench_isometry3_transform_point__alt_rotate_then_add` and
    /// `test_transform_point_fused_and_composed_agree_bit_for_bit`.
    ///
    /// Not an upstream method: upstream writes `rotation * v + translation`, which in fixed point
    /// is exactly this kernel. Panics on overflow of an intermediate doubling (`|v|` above about
    /// `2^30`).
    #[inline(always)]
    fn rotate_translate(r: UnitQuaternion<T>, v: Vector3<T>, t: Vector3<T>) -> Vector3<T> {
        let u = r.imag();
        let c = u.cross(v);
        let d = Vector3 { x: c.x + c.x, y: c.y + c.y, z: c.z + c.z };
        let uxd = u.cross(d);
        let w = r.quaternion.w;
        Vector3 {
            x: R::wide_rescale(
                R::wide_add(
                    R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, d.x), uxd.x), v.x),
                    t.x,
                ),
            ),
            y: R::wide_rescale(
                R::wide_add(
                    R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, d.y), uxd.y), v.y),
                    t.y,
                ),
            ),
            z: R::wide_rescale(
                R::wide_add(
                    R::wide_add(R::wide_add(R::wide_add_prod(R::wide_zero(), w, d.z), uxd.z), v.z),
                    t.z,
                ),
            ),
        }
    }
    /// Renormalises the rotation exactly (`UnitQuaternion::renormalize`: one norm and four exactly
    /// correctly rounded divisions), leaving the translation untouched. Panics with
    /// `Fixed: division by zero` on a zero rotation. Upstream: `Unit::renormalize` applied to the
    /// rotation part (upstream has no `Isometry::renormalize`).
    #[inline(always)]
    fn renormalize(self: Isometry3<T>) -> Isometry3<T> {
        let mut rotation = self.rotation;
        let _ = rotation.renormalize();
        Isometry3 { rotation, translation: self.translation }
    }
    /// Renormalises the rotation with one Newton step (`UnitQuaternion::renormalize_fast`: no
    /// square root, no division, 15 % cheaper than `renormalize`), for a norm already within about
    /// `2^-16` of 1 — which is what a pose composed every step drifts to. This is the call a
    /// rigid-body integrator makes once per body per step. Upstream: `Unit::renormalize_fast`
    /// applied to the rotation part.
    #[inline(always)]
    fn renormalize_fast(self: Isometry3<T>) -> Isometry3<T> {
        let mut rotation = self.rotation;
        rotation.renormalize_fast();
        Isometry3 { rotation, translation: self.translation }
    }
    /// Interpolation WITHOUT trigonometry: the translations linearly, the rotations by
    /// `UnitQuaternion::nlerp` (four fused lerps, one norm, four divisions). `t` is not clamped.
    ///
    /// Measured 31 610 gas against 91 650 for `lerp_slerp` (2.9x), which pays an `acos` and two
    /// `sin` (`bench_isometry3_lerp_slerp__*`). Like upstream's `nlerp` it does NOT take the
    /// shortest arc and its angular velocity is not constant (the chord is walked at constant
    /// speed): for the small relative rotations of one physics step the difference is far below an
    /// ulp, for rendering between two distant poses it is visible. Panics with
    /// `Fixed: division by zero` when the interpolated quaternion vanishes (exactly opposite
    /// rotations at `t = 1/2`).
    ///
    /// Upstream has no `Isometry3::lerp_nlerp`; this is `lerp_slerp` with `nlerp` in place of
    /// `slerp`.
    fn lerp_nlerp(self: Isometry3<T>, other: Isometry3<T>, t: T) -> Isometry3<T> {
        Isometry3 {
            rotation: self.rotation.nlerp(other.rotation, t),
            translation: Translation3 {
                vector: Vector3 {
                    x: R::lerp(self.translation.vector.x, other.translation.vector.x, t),
                    y: R::lerp(self.translation.vector.y, other.translation.vector.y, t),
                    z: R::lerp(self.translation.vector.z, other.translation.vector.z, t),
                },
            },
        }
    }
}
