//! In `nalgebra_static4`: the trait `Unit4Trait`; the impl `Unit4Impl`. This module is split over
//! packages; the other parts are in `nalgebra_core`, `nalgebra_static2`, `nalgebra_static3`.
//!
//! `Unit<V>`: a wrapper guaranteeing (by contract) that a vector has unit norm (upstream
//! `nalgebra::Unit`).
//!
//! - `Normed<V, T>`: the few operations `Unit` needs from a vector type `V` over the scalar
//!   `T` (upstream `Normed`), implemented for the column vectors `Vector1<T>` to `Vector6<T>`;
//! - `UnitTrait` / `UnitImpl`: construction (`new_normalize`, `try_new`, `new_unchecked`, ...),
//!   in-place renormalization (upstream's `&mut self` methods) and the products upstream reaches
//!   through `Deref`, generic over any `Normed` vector;
//! - `Unit2Trait` / `Unit3Trait` / `Unit4Trait`: the axes (`x_axis`, ...);
//! - `-u` (exact, a negated unit vector is a unit vector).
//!
//! The `value` field is public: unlike upstream there is no `Deref`, so vector operations are
//! reached through `u.value` (`u.value.cross(v)`), and `new_unchecked` is just `Unit { value }`.
//! Unit vectors built by `new_normalize` have a norm of `1` within a few ulp, not exactly `1`
//! (fixed point cannot do better); `renormalize` / `renormalize_fast` bring them back.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use nalgebra_core::base::unit::Unit;
use nalgebra_types4::base::vector4::Vector4;
use simba::scalar::Real;

/// The axes of 4-space as unit vectors. Upstream: `Vector4::x_axis`, ..., `Vector4::w_axis`.
pub trait Unit4Trait<T> {
    /// The unit axis `(1, 0, 0, 0)`. Exact. Upstream: `Vector4::x_axis`.
    fn x_axis() -> Unit<Vector4<T>>;
    /// The unit axis `(0, 1, 0, 0)`. Exact. Upstream: `Vector4::y_axis`.
    fn y_axis() -> Unit<Vector4<T>>;
    /// The unit axis `(0, 0, 1, 0)`. Exact. Upstream: `Vector4::z_axis`.
    fn z_axis() -> Unit<Vector4<T>>;
    /// The unit axis `(0, 0, 0, 1)`. Exact. Upstream: `Vector4::w_axis`.
    fn w_axis() -> Unit<Vector4<T>>;
}

pub impl Unit4Impl<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Unit4Trait<T> {
    #[inline(always)]
    fn x_axis() -> Unit<Vector4<T>> {
        Unit { value: Vector4 { x: R::one(), y: R::zero(), z: R::zero(), w: R::zero() } }
    }

    #[inline(always)]
    fn y_axis() -> Unit<Vector4<T>> {
        Unit { value: Vector4 { x: R::zero(), y: R::one(), z: R::zero(), w: R::zero() } }
    }

    #[inline(always)]
    fn z_axis() -> Unit<Vector4<T>> {
        Unit { value: Vector4 { x: R::zero(), y: R::zero(), z: R::one(), w: R::zero() } }
    }

    #[inline(always)]
    fn w_axis() -> Unit<Vector4<T>> {
        Unit { value: Vector4 { x: R::zero(), y: R::zero(), z: R::zero(), w: R::one() } }
    }
}
