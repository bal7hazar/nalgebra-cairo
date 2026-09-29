//! Internal, no stability promise: the crate-private items of `geometry::isometry2` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types2::geometry::translation2::Translation2;
use simba::scalar::Real;
use crate::geometry::isometry2::Isometry2;
use crate::geometry::unit_complex::{UnitComplex, UnitComplexTrait};

/// Crate-internal kernels of `Isometry2<T>` (WP 8.0: the public API is strictly upstream's): the
/// fused `rotate_translate` behind every "rotate then translate" (DESIGN D6), the renormalisation
/// of the rotation part (upstream renormalizes `iso.rotation` itself, in place) and the
/// trigonometry-free `lerp_nlerp` (upstream has `lerp_slerp` only).
#[generate_trait]
pub impl Isometry2InternalImpl<
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
