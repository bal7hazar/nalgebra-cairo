//! `Unit<V>`: a wrapper guaranteeing (by contract) that a vector has unit norm (upstream
//! `nalgebra::Unit`).
//!
//! - `Normalizable<V, T>`: the few operations `Unit` needs from a vector type `V` over the scalar
//!   `T`, implemented for `Vector2<T>`, `Vector3<T>` and `Vector4<T>`;
//! - `UnitTrait` / `UnitImpl`: construction (`new_normalize`, `try_new`, `new_unchecked`, ...),
//!   renormalization and products, generic over any `Normalizable` vector;
//! - `Unit2Trait` / `Unit3Trait` / `Unit4Trait`: the axes (`x_axis`, ...) and, in 3D, the
//!   orthonormal basis;
//! - `-u` (exact, a negated unit vector is a unit vector).
//!
//! The `value` field is public: unlike upstream there is no `Deref`, so vector operations are
//! reached through `u.value` (`u.value.cross(v)`), and `new_unchecked` is just `Unit { value }`.
//! Unit vectors built by `new_normalize` have a norm of `1` within a few ulp, not exactly `1`
//! (fixed point cannot do better); `renormalize` / `renormalize_fast` bring them back.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use simba::scalar::Real;
use super::vector2::Vector2;
use super::vector3::{Vector3, Vector3Trait};
use super::vector4::Vector4;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

/// A vector of unit norm. Nothing enforces the invariant: build it with `new_normalize` /
/// `try_new`, or with `new_unchecked` when the vector is known to be normalized.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Unit<V> {
    pub value: V,
}

/// What `Unit<V>` needs from a vector type `V` over the scalar `T` (implemented by `Vector2`,
/// `Vector3` and `Vector4`): norms, products and scaling, every sum of products fused.
pub trait Normalizable<V, T> {
    /// Euclidean norm, floored once, no intermediate overflow. Upstream: `norm`.
    fn norm(self: V) -> T;
    /// Squared Euclidean norm, fused (floored once). Panics on overflow. Upstream:
    /// `norm_squared`.
    fn norm_squared(self: V) -> T;
    /// `self * k`, each component floored once. Panics on overflow. Upstream: `scale`.
    fn scale(self: V, k: T) -> V;
    /// `self / k`, each component truncated toward zero. Panics on a zero `k` and on overflow.
    /// Upstream: `unscale`.
    fn unscale(self: V, k: T) -> V;
    /// Dot product, fused (floored once). Panics on overflow. Upstream: `dot`.
    fn dot(self: V, rhs: V) -> T;
    /// `true` when every component is within `ulps` raw units of `rhs`'s. Upstream: `abs_diff_eq`.
    fn abs_diff_eq(self: V, rhs: V, ulps: u64) -> bool;
}

pub impl Vector2Normalizable<
    T, impl R: Real<T>, +Mul<T>, +Copy<T>, +Drop<T>,
> of Normalizable<Vector2<T>, T> {
    #[inline(always)]
    fn norm(self: Vector2<T>) -> T {
        R::norm2(self.x, self.y)
    }

    #[inline(always)]
    fn norm_squared(self: Vector2<T>) -> T {
        R::norm_squared2(self.x, self.y)
    }

    #[inline(always)]
    fn scale(self: Vector2<T>, k: T) -> Vector2<T> {
        Vector2 { x: self.x * k, y: self.y * k }
    }

    #[inline(always)]
    fn unscale(self: Vector2<T>, k: T) -> Vector2<T> {
        Vector2 { x: R::div(self.x, k), y: R::div(self.y, k) }
    }

    #[inline(always)]
    fn dot(self: Vector2<T>, rhs: Vector2<T>) -> T {
        R::sum_prod2(self.x, rhs.x, self.y, rhs.y)
    }

    #[inline(always)]
    fn abs_diff_eq(self: Vector2<T>, rhs: Vector2<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, rhs.x, ulps) && R::abs_diff_eq(self.y, rhs.y, ulps)
    }
}

pub impl Vector3Normalizable<
    T, impl R: Real<T>, +Mul<T>, +Copy<T>, +Drop<T>,
> of Normalizable<Vector3<T>, T> {
    #[inline(always)]
    fn norm(self: Vector3<T>) -> T {
        R::norm3(self.x, self.y, self.z)
    }

    #[inline(always)]
    fn norm_squared(self: Vector3<T>) -> T {
        R::norm_squared3(self.x, self.y, self.z)
    }

    #[inline(always)]
    fn scale(self: Vector3<T>, k: T) -> Vector3<T> {
        Vector3 { x: self.x * k, y: self.y * k, z: self.z * k }
    }

    #[inline(always)]
    fn unscale(self: Vector3<T>, k: T) -> Vector3<T> {
        Vector3 { x: R::div(self.x, k), y: R::div(self.y, k), z: R::div(self.z, k) }
    }

    #[inline(always)]
    fn dot(self: Vector3<T>, rhs: Vector3<T>) -> T {
        R::sum_prod3(self.x, rhs.x, self.y, rhs.y, self.z, rhs.z)
    }

    #[inline(always)]
    fn abs_diff_eq(self: Vector3<T>, rhs: Vector3<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, rhs.x, ulps)
            && R::abs_diff_eq(self.y, rhs.y, ulps)
            && R::abs_diff_eq(self.z, rhs.z, ulps)
    }
}

pub impl Vector4Normalizable<
    T, impl R: Real<T>, +Mul<T>, +Copy<T>, +Drop<T>,
> of Normalizable<Vector4<T>, T> {
    #[inline(always)]
    fn norm(self: Vector4<T>) -> T {
        R::norm4(self.x, self.y, self.z, self.w)
    }

    #[inline(always)]
    fn norm_squared(self: Vector4<T>) -> T {
        R::norm_squared4(self.x, self.y, self.z, self.w)
    }

    #[inline(always)]
    fn scale(self: Vector4<T>, k: T) -> Vector4<T> {
        Vector4 { x: self.x * k, y: self.y * k, z: self.z * k, w: self.w * k }
    }

    #[inline(always)]
    fn unscale(self: Vector4<T>, k: T) -> Vector4<T> {
        Vector4 {
            x: R::div(self.x, k), y: R::div(self.y, k), z: R::div(self.z, k), w: R::div(self.w, k),
        }
    }

    #[inline(always)]
    fn dot(self: Vector4<T>, rhs: Vector4<T>) -> T {
        R::sum_prod4(self.x, rhs.x, self.y, rhs.y, self.z, rhs.z, self.w, rhs.w)
    }

    #[inline(always)]
    fn abs_diff_eq(self: Vector4<T>, rhs: Vector4<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.x, rhs.x, ulps)
            && R::abs_diff_eq(self.y, rhs.y, ulps)
            && R::abs_diff_eq(self.z, rhs.z, ulps)
            && R::abs_diff_eq(self.w, rhs.w, ulps)
    }
}

/// Operations of `Unit<V>` over a `Normalizable` vector type `V` with scalar `T`. By value,
/// unrolled, no loop.
pub trait UnitTrait<V, T> {
    /// Wraps `v` WITHOUT normalizing it: the caller guarantees a unit norm. Upstream:
    /// `Unit::new_unchecked` (equal to the struct literal `Unit { value: v }`).
    fn new_unchecked(v: V) -> Unit<V>;
    /// `v / |v|`: the norm (floored once), then one truncated division per component, so
    /// the error is about `1 + 1 / |v|` ulp per component whatever the magnitude of `v` (see
    /// `Vector3Trait::normalize`). Panics with `Fixed: division by zero` when the norm is zero,
    /// and on overflow when the norm does not fit. Upstream: `Unit::new_normalize`.
    fn new_normalize(v: V) -> Unit<V>;
    /// `(Unit::new_normalize(v), |v|)`: the unit vector and the norm it was divided by. Same
    /// panics. Upstream: `Unit::new_and_get`.
    fn new_and_get(v: V) -> (Unit<V>, T);
    /// `Some(Unit::new_normalize(v))`, or `None` when the norm is `<= min_norm`. With
    /// `min_norm >= 0` it never divides by zero. Upstream: `Unit::try_new`.
    fn try_new(v: V, min_norm: T) -> Option<Unit<V>>;
    /// `Some((Unit::new_normalize(v), |v|))`, or `None` when the norm is `<= min_norm`. Upstream:
    /// `Unit::try_new_and_get`.
    fn try_new_and_get(v: V, min_norm: T) -> Option<(Unit<V>, T)>;
    /// The wrapped vector. Upstream: `Unit::into_inner`.
    fn into_inner(self: Unit<V>) -> V;
    /// The wrapped vector (a copy: everything is by value here). Upstream: `Unit::as_ref`
    /// (`AsRef`), a reference in Rust.
    fn as_ref(self: Unit<V>) -> V;
    /// Renormalizes exactly: `Unit::new_normalize(self.value)`, i.e. one norm and one exactly
    /// truncated division per component. Panics on a zero norm. Upstream: `Unit::renormalize`
    /// (which also returns the previous norm and works in place).
    fn renormalize(self: Unit<V>) -> Unit<V>;
    /// Renormalizes a vector that already has a norm close to 1 (accumulated rounding errors,
    /// e.g. an axis rotated every step): one Newton step for the inverse square root,
    /// `v * (3 - |v|²) / 2`. One fused `norm_squared`, the factor `(3 - |v|²) / 2` as ONE fused
    /// kernel (`mul_add(|v|², -1/2, 3/2)`, floored once, bit-identical to upstream's
    /// `1/2 * (3 - |v|²)`), then one product per component. No square root, no division.
    /// Upstream: `Unit::renormalize_fast`.
    ///
    /// With `|v|² = 1 + e` the new squared norm is `1 - 3e²/4 + e³/4`: the error is squared at
    /// every step, so it converges for `|e| < 1` and is exact to the last ulp after one step for
    /// `|e| < 2^-16` (a step also floors, so the norm stays within about 2 ulp below 1). A zero
    /// vector stays zero (no panic). Panics on overflow when `|v|²` does not fit (above a norm
    /// of about 46 340).
    ///
    /// Measured on `Unit<Vector3<Fixed>>` (Sierra gas net of the baseline): this form 9 650,
    /// upstream's literal `HALF * (THREE - s)` 10 490, the exact `renormalize` 10 740 (a `Fixed`
    /// division costs about 2 800 and a product about 1 800, so dropping the divisions and the
    /// square root gains only about 10 %), a `mul_add` per component 11 090, `v * (1 + f)` 11 330,
    /// a `lerp` per component 12 590. All the variants give the same bits and are kept as
    /// benchmarks (`bench_unit3_renormalize_fast__alt_*`). Use `renormalize` where the norm may be
    /// far from 1.
    fn renormalize_fast(self: Unit<V>) -> Unit<V>;
    /// Dot product of two unit vectors: the cosine of the angle between them, fused (floored
    /// once). Cannot overflow (`|cos| <= 1` up to a few ulp). Upstream: `dot` (through `Deref`).
    fn dot(self: Unit<V>, rhs: Unit<V>) -> T;
    /// Dot product with any vector: the signed length of the projection of `rhs` on `self`,
    /// fused (floored once). Panics on overflow. Upstream: `dot` (through `Deref`).
    fn dot_vector(self: Unit<V>, rhs: V) -> T;
    /// `self * k`, a vector of norm `|k|` along `self`, each component floored once. Panics on
    /// overflow. Upstream: `Unit * k` (through `Deref`, e.g. the scaled axis of a rotation).
    fn scale(self: Unit<V>, k: T) -> V;
    /// `true` when every component is within `ulps` raw units of `other`'s; cannot overflow.
    /// Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of
    /// a float epsilon (DESIGN D3).
    fn abs_diff_eq(self: Unit<V>, other: Unit<V>, ulps: u64) -> bool;
}

pub impl UnitImpl<
    V,
    T,
    impl N: Normalizable<V, T>,
    impl R: Real<T>,
    +Add<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialOrd<T>,
    +Copy<V>,
    +Drop<V>,
    +Copy<T>,
    +Drop<T>,
> of UnitTrait<V, T> {
    #[inline(always)]
    fn new_unchecked(v: V) -> Unit<V> {
        Unit { value: v }
    }

    #[inline(always)]
    fn new_normalize(v: V) -> Unit<V> {
        Unit { value: N::unscale(v, N::norm(v)) }
    }

    #[inline(always)]
    fn new_and_get(v: V) -> (Unit<V>, T) {
        let n = N::norm(v);
        (Unit { value: N::unscale(v, n) }, n)
    }

    #[inline(always)]
    fn try_new(v: V, min_norm: T) -> Option<Unit<V>> {
        let n = N::norm(v);
        if n <= min_norm {
            None
        } else {
            Some(Unit { value: N::unscale(v, n) })
        }
    }

    #[inline(always)]
    fn try_new_and_get(v: V, min_norm: T) -> Option<(Unit<V>, T)> {
        let n = N::norm(v);
        if n <= min_norm {
            None
        } else {
            Some((Unit { value: N::unscale(v, n) }, n))
        }
    }

    #[inline(always)]
    fn into_inner(self: Unit<V>) -> V {
        self.value
    }

    #[inline(always)]
    fn as_ref(self: Unit<V>) -> V {
        self.value
    }

    #[inline(always)]
    fn renormalize(self: Unit<V>) -> Unit<V> {
        Unit { value: N::unscale(self.value, N::norm(self.value)) }
    }

    #[inline(always)]
    fn renormalize_fast(self: Unit<V>) -> Unit<V> {
        // (3 - |v|²) / 2 = floor(-|v|² * 1/2 + 3/2): one fused kernel, bit-identical to
        // `HALF * (THREE - s)`.
        let f = R::mul_add(N::norm_squared(self.value), -R::HALF, R::HALF + R::ONE);
        Unit { value: N::scale(self.value, f) }
    }

    #[inline(always)]
    fn dot(self: Unit<V>, rhs: Unit<V>) -> T {
        N::dot(self.value, rhs.value)
    }

    #[inline(always)]
    fn dot_vector(self: Unit<V>, rhs: V) -> T {
        N::dot(self.value, rhs)
    }

    #[inline(always)]
    fn scale(self: Unit<V>, k: T) -> V {
        N::scale(self.value, k)
    }

    #[inline(always)]
    fn abs_diff_eq(self: Unit<V>, other: Unit<V>, ulps: u64) -> bool {
        N::abs_diff_eq(self.value, other.value, ulps)
    }
}

/// `-u`: the opposite unit vector. Exact; panics on overflow (`-MIN`). Upstream: `Neg`.
pub impl UnitNeg<V, +Neg<V>, +Copy<V>, +Drop<V>> of Neg<Unit<V>> {
    #[inline(always)]
    fn neg(a: Unit<V>) -> Unit<V> {
        Unit { value: -a.value }
    }
}

/// The axes of the plane as unit vectors. Upstream: `Vector2::x_axis`, `Vector2::y_axis`.
pub trait Unit2Trait<T> {
    /// The unit axis `(1, 0)`. Exact. Upstream: `Vector2::x_axis`.
    fn x_axis() -> Unit<Vector2<T>>;
    /// The unit axis `(0, 1)`. Exact. Upstream: `Vector2::y_axis`.
    fn y_axis() -> Unit<Vector2<T>>;
}

pub impl Unit2Impl<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Unit2Trait<T> {
    #[inline(always)]
    fn x_axis() -> Unit<Vector2<T>> {
        Unit { value: Vector2 { x: R::ONE, y: R::ZERO } }
    }

    #[inline(always)]
    fn y_axis() -> Unit<Vector2<T>> {
        Unit { value: Vector2 { x: R::ZERO, y: R::ONE } }
    }
}

/// The axes of space as unit vectors, and the orthonormal basis of a unit vector. Upstream:
/// `Vector3::x_axis`, ..., `Vector3::orthonormal_subspace_basis`.
pub trait Unit3Trait<T> {
    /// The unit axis `(1, 0, 0)`. Exact. Upstream: `Vector3::x_axis`.
    fn x_axis() -> Unit<Vector3<T>>;
    /// The unit axis `(0, 1, 0)`. Exact. Upstream: `Vector3::y_axis`.
    fn y_axis() -> Unit<Vector3<T>>;
    /// The unit axis `(0, 0, 1)`. Exact. Upstream: `Vector3::z_axis`.
    fn z_axis() -> Unit<Vector3<T>>;
    /// Two unit vectors `(u, w)` orthogonal to `self` and to each other, with `u x w = self`,
    /// each within about 3 ulp: `Vector3Trait::orthonormal_basis` of the wrapped vector (the
    /// invariant that the vector is unit is the one of `Unit`). Upstream:
    /// `Vector3::orthonormal_subspace_basis(&[v], ..)`.
    fn orthonormal_basis(self: Unit<Vector3<T>>) -> (Unit<Vector3<T>>, Unit<Vector3<T>>);
}

pub impl Unit3Impl<
    T,
    impl R: Real<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Copy<T>,
    +Drop<T>,
> of Unit3Trait<T> {
    #[inline(always)]
    fn x_axis() -> Unit<Vector3<T>> {
        Unit { value: Vector3 { x: R::ONE, y: R::ZERO, z: R::ZERO } }
    }

    #[inline(always)]
    fn y_axis() -> Unit<Vector3<T>> {
        Unit { value: Vector3 { x: R::ZERO, y: R::ONE, z: R::ZERO } }
    }

    #[inline(always)]
    fn z_axis() -> Unit<Vector3<T>> {
        Unit { value: Vector3 { x: R::ZERO, y: R::ZERO, z: R::ONE } }
    }

    #[inline(always)]
    fn orthonormal_basis(self: Unit<Vector3<T>>) -> (Unit<Vector3<T>>, Unit<Vector3<T>>) {
        let (u, w) = Vector3Trait::<T>::orthonormal_basis(self.value);
        (Unit { value: u }, Unit { value: w })
    }
}

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
        Unit { value: Vector4 { x: R::ONE, y: R::ZERO, z: R::ZERO, w: R::ZERO } }
    }

    #[inline(always)]
    fn y_axis() -> Unit<Vector4<T>> {
        Unit { value: Vector4 { x: R::ZERO, y: R::ONE, z: R::ZERO, w: R::ZERO } }
    }

    #[inline(always)]
    fn z_axis() -> Unit<Vector4<T>> {
        Unit { value: Vector4 { x: R::ZERO, y: R::ZERO, z: R::ONE, w: R::ZERO } }
    }

    #[inline(always)]
    fn w_axis() -> Unit<Vector4<T>> {
        Unit { value: Vector4 { x: R::ZERO, y: R::ZERO, z: R::ZERO, w: R::ONE } }
    }
}
