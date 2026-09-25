//! `Quaternion`: a general quaternion `w + x·i + y·j + z·k` (upstream `nalgebra::Quaternion`).
//!
//! - `QuaternionTrait` / `QuaternionImpl`: constructors, parts, algebra, norms, inverse and
//!   interpolation, generic over a `simba::scalar::Real` scalar;
//! - operators `+`, `-`, unary `-` and `*` (the Hamilton product) and the conversions from / to
//!   `Vector4<T>`: their impls live in this module, where the compiler finds them without any
//!   import.
//!
//! Conventions (DESIGN D8, pinned by `test_new_is_w_first` and `test_serde_is_imag_first`):
//!
//! | | order |
//! |---|---|
//! | `new` arguments | `(w, i, j, k)`, like upstream `Quaternion::new` |
//! | fields / `Serde` / `as_vector` | `(i, j, k, w)`, like upstream's `coords: Vector4` |
//! | glam, for glam-cairo's conversions | `Quat::from_xyzw(x, y, z, w)` = `(i, j, k, w)` |
//!
//! The unit quaternion of a 3D rotation is `UnitQuaternion` (`geometry::unit_quaternion`); this
//! type is the general algebra it is built on.
//!
//! - `QuaternionTranscendentalTrait` / `QuaternionTranscendentalImpl`: the transcendental
//!   functions of the general algebra (`exp`, `ln`, `powf`, the trigonometric and hyperbolic
//!   families, the polar decomposition), which additionally need `simba::scalar::Transcendental`
//!   (WP 8.4-P08). They cost one to several transcendental calls each (30 000 to 300 000 gas):
//!   nothing in the physics stack calls them — `UnitQuaternion::from_scaled_axis` IS `exp` of a
//!   pure quaternion and `scaled_axis` its `ln`, both cheaper.
//!
//! The approximate comparisons (`abs_diff_eq`, `relative_eq`, `ulps_eq`) count their tolerances
//! in ulp (DESIGN D3) and, like upstream's `approx` impls, accept `other` OR `-other`
//! component-wise (the double cover of the rotations: `q` and `-q` compare equal).
//!
//! Where upstream's formula yields `NaN` (the logarithm, square root and inverse trigonometric
//! functions of a REAL quaternion, whose imaginary part upstream normalises), these functions
//! panic with `errors::REAL_QUATERNION` instead (PLAN M8 fidelity rules).
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use core::num::traits::{One, Zero};
use core::ops::Index;
use simba::scalar::{Real, Transcendental};
use crate::base::unit::{Unit, UnitTrait};
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod oracle_ext;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_ext;

/// Panic messages of the quaternion algebra (stable API).
pub mod errors {
    /// `tan`, `tanh` or `atan` divided by a quaternion whose squared norm floors to zero (upstream
    /// unwraps `None` there).
    pub const NOT_INVERTIBLE: felt252 = 'nalgebra: not invertible';
    /// `q[i]` with `i > 3`.
    pub const INDEX_OUT_OF_BOUNDS: felt252 = 'nalgebra: index out of bounds';
    /// `ln`, `sqrt`, `powf`, `acos`, `asin`, `atan`, `asinh`, `acosh` or `atanh` of an argument
    /// whose imaginary part is zero where upstream normalises it (its result is `NaN` there).
    pub const REAL_QUATERNION: felt252 = 'nalgebra: real quaternion (NaN)';
}

/// A quaternion `w + i·i + j·j + k·k`.
///
/// Fields are declared in the storage order of upstream's `coords` vector (`i, j, k, w`), so
/// `Serde` matches upstream, while `new` takes its arguments in upstream's `(w, i, j, k)` order.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Quaternion<T> {
    pub i: T,
    pub j: T,
    pub k: T,
    pub w: T,
}

/// Methods of `Quaternion<T>` for any `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl QuaternionImpl<
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
> of QuaternionTrait<T> {
    // --- constructors ------------------------------------------------------------------------

    /// The quaternion `w + x·i + y·j + z·k`, arguments in upstream's order. Upstream:
    /// `Quaternion::new`.
    #[inline(always)]
    fn new(w: T, i: T, j: T, k: T) -> Quaternion<T> {
        Quaternion { i, j, k, w }
    }

    /// The multiplicative identity `1`. Upstream: `Quaternion::identity`.
    #[inline(always)]
    fn identity() -> Quaternion<T> {
        Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::one() }
    }

    /// The quaternion of real part `scalar` and imaginary part `vector`. Upstream:
    /// `Quaternion::from_parts`.
    #[inline(always)]
    fn from_parts(scalar: T, vector: Vector3<T>) -> Quaternion<T> {
        Quaternion { i: vector.x, j: vector.y, k: vector.z, w: scalar }
    }

    /// The pure quaternion `(0, vector)`. Upstream: `Quaternion::from_imag`.
    #[inline(always)]
    fn from_imag(vector: Vector3<T>) -> Quaternion<T> {
        Quaternion { i: vector.x, j: vector.y, k: vector.z, w: R::zero() }
    }

    /// The real quaternion `(w, 0, 0, 0)`. Upstream: `Quaternion::from_real`.
    #[inline(always)]
    fn from_real(w: T) -> Quaternion<T> {
        Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w }
    }

    /// The quaternion whose coordinates are `(i, j, k, w)`, the STORAGE order. Upstream:
    /// `Quaternion::from_vector` (`From<Vector4>`).
    #[inline(always)]
    fn from_vector(coords: Vector4<T>) -> Quaternion<T> {
        Quaternion { i: coords.x, j: coords.y, k: coords.z, w: coords.w }
    }

    // --- parts -------------------------------------------------------------------------------

    /// The coordinates `(i, j, k, w)`, the STORAGE order (not `new`'s). Upstream: the `coords`
    /// field (`as_vector`).
    #[inline(always)]
    fn as_vector(self: Quaternion<T>) -> Vector4<T> {
        Vector4 { x: self.i, y: self.j, z: self.k, w: self.w }
    }

    /// The real part. Upstream: `scalar` (and the `w` field).
    #[inline(always)]
    fn scalar(self: Quaternion<T>) -> T {
        self.w
    }

    /// The imaginary part `(i, j, k)`. Upstream: `vector`.
    #[inline(always)]
    fn vector(self: Quaternion<T>) -> Vector3<T> {
        Vector3 { x: self.i, y: self.j, z: self.k }
    }

    /// Alias of `vector`. Upstream: `imag`.
    #[inline(always)]
    fn imag(self: Quaternion<T>) -> Vector3<T> {
        Vector3 { x: self.i, y: self.j, z: self.k }
    }

    // --- algebra -----------------------------------------------------------------------------

    /// `self * k`, each component floored once. Panics on overflow. Upstream: `Mul<T>`
    /// (`q * k`).
    #[inline(always)]
    fn scale(self: Quaternion<T>, k: T) -> Quaternion<T> {
        Quaternion { i: self.i * k, j: self.j * k, k: self.k * k, w: self.w * k }
    }

    /// `self / k`, each component being the correctly rounded quotient. Panics on a zero `k` and on
    /// overflow. Upstream: `Div<T>` (`q / k`).
    ///
    /// One division per component on purpose (see `Vector3Trait::unscale`): multiplying by the
    /// rounded reciprocal of `k` is cheaper but costs up to `|self|` ulp instead of 1.
    #[inline(always)]
    fn unscale(self: Quaternion<T>, k: T) -> Quaternion<T> {
        let (i, j, k, w) = R::div4(self.i, self.j, self.k, self.w, k);
        Quaternion { i, j, k, w }
    }

    /// `(w, -i, -j, -k)`. Exact; panics on overflow (`-MIN`). Upstream: `conjugate`.
    #[inline(always)]
    fn conjugate(self: Quaternion<T>) -> Quaternion<T> {
        Quaternion { i: -self.i, j: -self.j, k: -self.k, w: self.w }
    }

    /// `|self|²`, fused (floored once). Panics on overflow: above a norm of about 46 340 (Q32.32)
    /// only `norm` works. Upstream: `norm_squared` (`magnitude_squared`).
    #[inline(always)]
    fn norm_squared(self: Quaternion<T>) -> T {
        R::norm_squared4(self.i, self.j, self.k, self.w)
    }

    /// `|self|` (`Real::norm4`): square root of the UNSCALED exact sum of squares, floored once.
    /// No intermediate overflow: only the result must fit. Upstream: `norm` (`magnitude`).
    #[inline(always)]
    fn norm(self: Quaternion<T>) -> T {
        R::norm4(self.i, self.j, self.k, self.w)
    }

    /// Dot product of the coordinates, fused (floored once). Panics on overflow. Upstream: `dot`.
    #[inline(always)]
    fn dot(self: Quaternion<T>, rhs: Quaternion<T>) -> T {
        R::sum_prod4(self.i, rhs.i, self.j, rhs.j, self.k, rhs.k, self.w, rhs.w)
    }

    /// `self / |self|`: the floored norm, then one correctly rounded division per component, so the
    /// error is about `1 + 1 / |self|` ulp per component whatever the magnitude of `self` (see
    /// `Vector3Trait::normalize`). Panics with `Fixed: division by zero` on a zero quaternion.
    /// Upstream: `normalize`.
    #[inline(always)]
    fn normalize(self: Quaternion<T>) -> Quaternion<T> {
        Self::unscale(self, R::norm4(self.i, self.j, self.k, self.w))
    }

    /// `self⁻¹ = conjugate / |self|²`, or `None` when `|self|²` floors to zero (upstream
    /// compares it to zero with `relative_eq`). One correctly rounded division per component, so
    /// the error is about `1 + |q| / |q|²` ulp. Panics on overflow of `|self|²` (norm above about
    /// 46 340) and on `-MIN`. Upstream: `try_inverse`.
    #[inline(always)]
    fn try_inverse(self: Quaternion<T>) -> Option<Quaternion<T>> {
        let n2 = R::norm_squared4(self.i, self.j, self.k, self.w);
        if n2 == R::zero() {
            None
        } else {
            Some(
                {
                    let (i, j, k, w) = R::div4(-self.i, -self.j, -self.k, self.w, n2);
                    Quaternion { i, j, k, w }
                },
            )
        }
    }

    /// `self + (rhs - self) * t` per component (`Real::lerp`: exact difference and product, one
    /// floor rounding). `t` is not clamped; `t = 0` gives `self` and `t = 1` gives `rhs` exactly.
    /// The result is NOT a unit quaternion even when both inputs are (see
    /// `UnitQuaternionTrait::nlerp`). Panics on overflow. Upstream: `lerp`
    /// (`self * (1 - t) + rhs * t`).
    #[inline(always)]
    fn lerp(self: Quaternion<T>, rhs: Quaternion<T>, t: T) -> Quaternion<T> {
        Quaternion {
            i: R::lerp(self.i, rhs.i, t),
            j: R::lerp(self.j, rhs.j, t),
            k: R::lerp(self.k, rhs.k, t),
            w: R::lerp(self.w, rhs.w, t),
        }
    }

    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// the matching component of `other`, or every component within `ulps` of the matching
    /// component of `-other`: like upstream, `q` and `-q` compare equal (the double cover of the
    /// rotations). The second comparison runs only when the first fails; it negates `other`,
    /// hence panics on a component equal to the scalar's `MIN` there. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float
    /// epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Quaternion<T>, other: Quaternion<T>, ulps: u64) -> bool {
        (R::abs_diff_eq(self.i, other.i, ulps)
            && R::abs_diff_eq(self.j, other.j, ulps)
            && R::abs_diff_eq(self.k, other.k, ulps)
            && R::abs_diff_eq(self.w, other.w, ulps))
            || (R::abs_diff_eq(self.i, -other.i, ulps)
                && R::abs_diff_eq(self.j, -other.j, ulps)
                && R::abs_diff_eq(self.k, -other.k, ulps)
                && R::abs_diff_eq(self.w, -other.w, ulps))
    }

    /// `true` when every component is `abs_diff_eq` within `epsilon` ulp of `other`'s, or within
    /// `max_relative` times the larger magnitude of the two (`|a - b| <= max(|a|, |b|) ·
    /// max_relative`) — or the same against `-other` (upstream accepts `-q`, the double cover).
    /// Two components of opposite signs are compared absolutely only (their difference is at
    /// least the larger magnitude, so they are relatively equal only for `max_relative >= 1`,
    /// which is meaningless). Panics on a component equal to the scalar's `MIN`, and on overflow
    /// of `max(|a|, |b|) · max_relative` (only possible with `max_relative > 1`). Upstream:
    /// `approx::RelativeEq::relative_eq`, `epsilon` counted in ulp instead of a float epsilon
    /// (DESIGN D3).
    #[inline(always)]
    fn relative_eq(
        self: Quaternion<T>, other: Quaternion<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        (ApproxEqTrait::relative_eq(self.i, other.i, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.j, other.j, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.k, other.k, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(self.w, other.w, epsilon, max_relative))
            || (ApproxEqTrait::relative_eq(self.i, -other.i, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(self.j, -other.j, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(self.k, -other.k, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(self.w, -other.w, epsilon, max_relative))
    }

    /// `true` when every component is `abs_diff_eq` within `epsilon` ulp of `other`'s, or has the
    /// same sign and lies within `max_ulps` ulp — or the same against `-other` (upstream accepts
    /// `-q`, the double cover). In fixed point the distance in ulp IS the raw difference, so this
    /// is `abs_diff_eq` with the larger of the two budgets, except that the `max_ulps` budget
    /// does not cross zero (like upstream's float `ulps_eq`, which never compares the bits of
    /// values of opposite signs). The comparison against `-other` runs only when the first one
    /// fails and panics on a component equal to the scalar's `MIN`. Upstream:
    /// `approx::UlpsEq::ulps_eq`.
    #[inline(always)]
    fn ulps_eq(self: Quaternion<T>, other: Quaternion<T>, epsilon: u64, max_ulps: u32) -> bool {
        (ApproxEqTrait::ulps_eq(self.i, other.i, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.j, other.j, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.k, other.k, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(self.w, other.w, epsilon, max_ulps))
            || (ApproxEqTrait::ulps_eq(self.i, -other.i, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(self.j, -other.j, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(self.k, -other.k, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(self.w, -other.w, epsilon, max_ulps))
    }

    // --- P08 completion: norms, parts, casts --------------------------------------------------

    /// Alias of `norm`. Upstream: `magnitude`.
    #[inline(always)]
    fn magnitude(self: Quaternion<T>) -> T {
        R::norm4(self.i, self.j, self.k, self.w)
    }

    /// Alias of `norm_squared`. Upstream: `magnitude_squared`.
    #[inline(always)]
    fn magnitude_squared(self: Quaternion<T>) -> T {
        R::norm_squared4(self.i, self.j, self.k, self.w)
    }

    /// `true` when the real part is exactly zero. Upstream: `is_pure`.
    #[inline(always)]
    fn is_pure(self: Quaternion<T>) -> bool {
        self.w == R::zero()
    }

    /// The pure quaternion `(0, imag)`: the real part dropped. Exact. Upstream: `pure`.
    #[inline(always)]
    fn pure(self: Quaternion<T>) -> Quaternion<T> {
        Quaternion { i: self.i, j: self.j, k: self.k, w: R::zero() }
    }

    /// The same quaternion with every component converted by `Into<T, U>`. With the single scalar
    /// of this library (`Fixed`) it is the identity; it exists for scalar-generic code. Upstream:
    /// `cast` (and `SubsetOf<Quaternion<U>>`, the `nalgebra::convert` it goes through).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: Quaternion<T>) -> Quaternion<U> {
        Quaternion { i: self.i.into(), j: self.j.into(), k: self.k.into(), w: self.w.into() }
    }

    // --- P08 completion: algebra ---------------------------------------------------------------

    /// `self / 2`, each component the correctly rounded quotient (nearest, ties to even, like
    /// upstream's `self / 2.0`): one `div4`, 11 270 gas. Multiplying by `1/2` instead costs 6 620
    /// but floors the odd raw values, a different result from upstream's `/ 2`
    /// (`bench_quaternion_half__alt_scale_half`, `test_half_alt_scale_half_floors_odd_raws`).
    /// Cannot overflow. Upstream: `half`.
    #[inline(always)]
    fn half(self: Quaternion<T>) -> Quaternion<T> {
        let (i, j, k, w) = R::div4(self.i, self.j, self.k, self.w, R::TWO);
        Quaternion { i, j, k, w }
    }

    /// `self * self` in its reduced form `(w² - |v|², 2w·v)` (the cross product of `v` with
    /// itself vanishes): one fused kernel per component, 10 products instead of the 16 of the
    /// Hamilton product, each component the exactly floored square — the same bits as `self *
    /// self`, 9 950 gas against 11 550 (`test_squared_matches_the_hamilton_product`,
    /// `bench_quaternion_squared__alt_mul`).
    /// Panics on overflow of a component. Upstream: `squared`.
    fn squared(self: Quaternion<T>) -> Quaternion<T> {
        let Quaternion { i, j, k, w } = self;
        let ww = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), w, w), i, i);
        let ww = R::wide_rescale(R::wide_sub_prod(R::wide_sub_prod(ww, j, j), k, k));
        Quaternion {
            i: R::wide_rescale(R::wide_add_prod(R::wide_add_prod(R::wide_zero(), w, i), w, i)),
            j: R::wide_rescale(R::wide_add_prod(R::wide_add_prod(R::wide_zero(), w, j), w, j)),
            k: R::wide_rescale(R::wide_add_prod(R::wide_add_prod(R::wide_zero(), w, k), w, k)),
            w: ww,
        }
    }

    /// The symmetric part of the product, `(self * other + other * self) / 2`, in its reduced
    /// form `(w₁w₂ - v₁·v₂, w₁v₂ + w₂v₁)` (the cross products cancel): one fused
    /// kernel per component, each the exactly floored result, where upstream rounds two Hamilton
    /// products and halves: 10 350 gas against 44 000 (`bench_quaternion_inner__alt_products`,
    /// within 1 ulp: `test_inner_alt_products_agrees`). Panics on overflow. Upstream: `inner`.
    fn inner(self: Quaternion<T>, other: Quaternion<T>) -> Quaternion<T> {
        let w = R::wide_sub_prod(
            R::wide_add_prod(R::wide_zero(), self.w, other.w), self.i, other.i,
        );
        let w = R::wide_rescale(
            R::wide_sub_prod(R::wide_sub_prod(w, self.j, other.j), self.k, other.k),
        );
        Quaternion {
            i: R::sum_prod2(self.w, other.i, other.w, self.i),
            j: R::sum_prod2(self.w, other.j, other.w, self.j),
            k: R::sum_prod2(self.w, other.k, other.w, self.k),
            w,
        }
    }

    /// The antisymmetric part of the product, `(self * other - other * self) / 2 = (0, v₁ ×
    /// v₂)`:
    /// one `diff_prod` per imaginary component (exactly floored), the real part exactly zero.
    /// Panics on overflow. Upstream: `outer`.
    #[inline(always)]
    fn outer(self: Quaternion<T>, other: Quaternion<T>) -> Quaternion<T> {
        Quaternion {
            i: R::diff_prod(self.j, other.k, self.k, other.j),
            j: R::diff_prod(self.k, other.i, self.i, other.k),
            k: R::diff_prod(self.i, other.j, self.j, other.i),
            w: R::zero(),
        }
    }

    /// `self * other⁻¹`, or `None` when `|other|²` floors to zero (like `try_inverse`).
    /// Computed as `(self * conj(other)) / |other|²`: one fused Hamilton product with the
    /// conjugate's signs folded in (floored once per component), then one correctly rounded
    /// division per component — never the rounded inverse, whose half-ulp errors upstream's
    /// `self * other⁻¹` multiplies by `|self|`: 1 ulp against 2 500 on a quotient of norm ~1 300
    /// (`test_right_div_alt_inverse_then_mul_loses_bits_on_large_quotients`). The price: 34 020
    /// gas against 27 630 for the loser on the bench inputs (the divisions of the larger
    /// numerators are dearer, `bench_quaternion_right_div__alt_inverse_then_mul`). Panics on
    /// overflow (of `|other|²` above a norm of about 46 340, or of the product). Upstream:
    /// `right_div`.
    fn right_div(self: Quaternion<T>, other: Quaternion<T>) -> Option<Quaternion<T>> {
        let n2 = R::norm_squared4(other.i, other.j, other.k, other.w);
        if n2 == R::zero() {
            return None;
        }
        let p = QuaternionInternalTrait::mul_conj(self, other);
        let (i, j, k, w) = R::div4(p.i, p.j, p.k, p.w, n2);
        Some(Quaternion { i, j, k, w })
    }

    /// `other⁻¹ * self`, or `None` when `|other|²` floors to zero: `(conj(other) * self) /
    /// |other|²`, the fused `conj_mul` then one correctly rounded division per component (see
    /// `right_div`). Upstream: `left_div`.
    fn left_div(self: Quaternion<T>, other: Quaternion<T>) -> Option<Quaternion<T>> {
        let n2 = R::norm_squared4(other.i, other.j, other.k, other.w);
        if n2 == R::zero() {
            return None;
        }
        let p = other.conj_mul(self);
        let (i, j, k, w) = R::div4(p.i, p.j, p.k, p.w, n2);
        Some(Quaternion { i, j, k, w })
    }

    /// The component of `self` along `other`: `inner(self, other).right_div(other)`, or `None`
    /// when `other` is not invertible. Upstream: `project`.
    #[inline(always)]
    fn project(self: Quaternion<T>, other: Quaternion<T>) -> Option<Quaternion<T>> {
        Self::right_div(Self::inner(self, other), other)
    }

    /// The component of `self` orthogonal to `other`: `outer(self, other).right_div(other)`, or
    /// `None` when `other` is not invertible. Upstream: `reject`.
    #[inline(always)]
    fn reject(self: Quaternion<T>, other: Quaternion<T>) -> Option<Quaternion<T>> {
        Self::right_div(Self::outer(self, other), other)
    }

    /// The principal square root (the root of non-negative real part), ALGEBRAICALLY:
    /// with `n = |self|`,
    /// - `w >= 0`: `s = sqrt((n + w) / 2)`, root `(s, v / 2s)`;
    /// - `w < 0`: `t = sqrt((n - w) / 2)` (no cancellation), root `(|v| / 2t, t · v / |v|)`, the
    ///   imaginary part as the correctly rounded quotients of the floored `v_i · t` by `|v|`
    ///   while `n < 2^20` (about one ulp), as `t` times the rounded direction above (a relative
    ///   error of about 2^-33);
    /// - a REAL quaternion (`v = 0`, zero included) panics with `errors::REAL_QUATERNION`:
    ///   upstream's `powf(1/2)` goes through `ln`, which normalises the zero imaginary part and
    ///   returns `NaN` there (even for a positive real, whose root would be obvious; PLAN M8
    ///   fidelity rules).
    ///
    /// `(n ± w) / 2` is one fused kernel (exactly floored, no overflow of the intermediate sum),
    /// then one square root and at most one norm and four divisions: 35 510 gas, where upstream's
    /// `powf(1/2)` = `exp(ln(q) / 2)` costs a `ln`, an `atan2`, an `exp` and a `sin_cos`: 148 110
    /// (`bench_quaternion_sqrt__alt_powf`, 4.2x dearer, and less accurate: the exponential
    /// amplifies the error of the logarithm). The two agree to the tolerance of the oracle
    /// (`test_sqrt_alt_powf_agrees`). Upstream: `sqrt`.
    fn sqrt(self: Quaternion<T>) -> Quaternion<T> {
        let Quaternion { i, j, k, w } = self;
        let nv = R::norm3(i, j, k);
        if nv == R::zero() {
            core::panic_with_felt252(errors::REAL_QUATERNION);
        }
        let n = R::norm4(i, j, k, w);
        if w >= R::zero() {
            let h = R::wide_add_prod(R::wide_add_prod(R::wide_zero(), n, R::HALF), w, R::HALF);
            let s = R::sqrt(R::wide_rescale(h));
            let (x, y, z) = R::div3(i, j, k, s + s);
            Quaternion { i: x, j: y, k: z, w: s }
        } else {
            let h = R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), n, R::HALF), w, R::HALF);
            let t = R::sqrt(R::wide_rescale(h));
            // t · v / |v|: dividing the floored products `v_i · t` keeps the error at about one
            // ulp; `v_i · t` fits while `|self| < 2^20` (`t <= 2^10`). Beyond that the unit
            // direction is scaled instead, which keeps a RELATIVE error of about 2^-33 (an
            // absolute error of up to `t / 2` ulp).
            let (x, y, z) = if n < R::from_int(0x100000) {
                R::div3(i * t, j * t, k * t, nv)
            } else {
                let (x, y, z) = R::div3(i, j, k, nv);
                (x * t, y * t, z * t)
            };
            Quaternion { i: x, j: y, k: z, w: R::div(nv, t + t) }
        }
    }
}

/// Crate-internal kernels of `Quaternion<T>` (WP 8.0: the public API is strictly upstream's): the
/// fused `a.conjugate() * b` of `UnitQuaternion::conj_mul` / `Isometry3::inv_mul` (WP 4.5).
#[generate_trait]
pub(crate) impl QuaternionInternalImpl<
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
> of QuaternionInternalTrait<T> {
    /// `self.conjugate() * other`, the Hamilton product with the conjugate on the LEFT, as ONE
    /// fused kernel: the three minus signs of the conjugate are folded into the accumulation
    /// (`Real::wide_add_prod` / `Real::wide_sub_prod` swapped where `self`'s imaginary part
    /// enters), in the term order of `QuaternionMul`. Since `(-a)·b = -(a·b)` exactly in the wide
    /// accumulator, the exact sums are the same and the result is bit-identical to
    /// `self.conjugate() * other` — 16 products, 4 roundings, no negation: 11 860 gas against
    /// 12 460 for `conjugate()` then `*` (`bench_quaternion_conj_mul__*`).
    ///
    /// The only behavioural difference: a component of `self` equal to the scalar's `MIN` no
    /// longer panics (the conjugate would have negated it); only an overflow of a result
    /// component panics (`Fixed: overflow`). Upstream has no direct equivalent: it replaces
    /// `q.conjugate() * other` (and `q.try_inverse().unwrap() * other` for a unit `q`), the
    /// rotation part of `Isometry3::inv_mul`.
    fn conj_mul(self: Quaternion<T>, other: Quaternion<T>) -> Quaternion<T> {
        // w = aw·bw + ai·bi + aj·bj + ak·bk
        let w = R::wide_add_prod(R::wide_zero(), self.w, other.w);
        let w = R::wide_add_prod(R::wide_add_prod(w, self.i, other.i), self.j, other.j);
        let w = R::wide_rescale(R::wide_add_prod(w, self.k, other.k));
        // i = aw·bi - ai·bw - aj·bk + ak·bj
        let i = R::wide_add_prod(R::wide_zero(), self.w, other.i);
        let i = R::wide_sub_prod(R::wide_sub_prod(i, self.i, other.w), self.j, other.k);
        let i = R::wide_rescale(R::wide_add_prod(i, self.k, other.j));
        // j = aw·bj + ai·bk - aj·bw - ak·bi
        let j = R::wide_add_prod(R::wide_zero(), self.w, other.j);
        let j = R::wide_sub_prod(R::wide_add_prod(j, self.i, other.k), self.j, other.w);
        let j = R::wide_rescale(R::wide_sub_prod(j, self.k, other.i));
        // k = aw·bk - ai·bj + aj·bi - ak·bw
        let k = R::wide_add_prod(R::wide_zero(), self.w, other.k);
        let k = R::wide_add_prod(R::wide_sub_prod(k, self.i, other.j), self.j, other.i);
        let k = R::wide_rescale(R::wide_sub_prod(k, self.k, other.w));
        Quaternion { i, j, k, w }
    }

    /// `self * other.conjugate()`, the Hamilton product with the conjugate on the RIGHT, as ONE
    /// fused kernel with the conjugate's three minus signs folded into the accumulation (the
    /// mirror of `conj_mul`): bit-identical to `self * other.conjugate()`, 16 products, 4
    /// roundings, no negation. The kernel of `right_div` and of `UnitQuaternion`'s `/`.
    fn mul_conj(self: Quaternion<T>, other: Quaternion<T>) -> Quaternion<T> {
        // w = aw·bw + ai·bi + aj·bj + ak·bk
        let w = R::wide_add_prod(R::wide_zero(), self.w, other.w);
        let w = R::wide_add_prod(R::wide_add_prod(w, self.i, other.i), self.j, other.j);
        let w = R::wide_rescale(R::wide_add_prod(w, self.k, other.k));
        // i = -aw·bi + ai·bw - aj·bk + ak·bj
        let i = R::wide_sub_prod(R::wide_zero(), self.w, other.i);
        let i = R::wide_sub_prod(R::wide_add_prod(i, self.i, other.w), self.j, other.k);
        let i = R::wide_rescale(R::wide_add_prod(i, self.k, other.j));
        // j = -aw·bj + ai·bk + aj·bw - ak·bi
        let j = R::wide_sub_prod(R::wide_zero(), self.w, other.j);
        let j = R::wide_add_prod(R::wide_add_prod(j, self.i, other.k), self.j, other.w);
        let j = R::wide_rescale(R::wide_sub_prod(j, self.k, other.i));
        // k = -aw·bk - ai·bj + aj·bi + ak·bw
        let k = R::wide_sub_prod(R::wide_zero(), self.w, other.k);
        let k = R::wide_add_prod(R::wide_sub_prod(k, self.i, other.j), self.j, other.i);
        let k = R::wide_rescale(R::wide_add_prod(k, self.k, other.w));
        Quaternion { i, j, k, w }
    }
}

/// Crate-internal scalar forms of upstream's `approx::RelativeEq` / `approx::UlpsEq`, shared by
/// the `relative_eq` / `ulps_eq` of `Quaternion`, `UnitQuaternion` and `UnitComplex` (tolerances
/// in ulp, DESIGN D3). See `QuaternionTrait::relative_eq` / `ulps_eq` for the semantics.
#[generate_trait]
pub(crate) impl ApproxEqImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Sub<T>, +Mul<T>, +PartialEq<T>,
> of ApproxEqTrait<T> {
    /// `|a - b| <= epsilon` ulp, or same signs and `|a - b| <= max(|a|, |b|) · max_relative`.
    #[inline(always)]
    fn relative_eq(a: T, b: T, epsilon: u64, max_relative: T) -> bool {
        if R::abs_diff_eq(a, b, epsilon) {
            return true;
        }
        if R::is_sign_negative(a) != R::is_sign_negative(b) {
            return false;
        }
        // `|a - b| <= limit`, with `Real` comparisons only (no `PartialOrd` needed by callers).
        let limit = R::max(R::abs(a), R::abs(b)) * max_relative;
        R::max(R::abs(a - b), limit) == limit
    }

    /// `|a - b| <= epsilon` ulp, or same signs and `|a - b| <= max_ulps` ulp.
    #[inline(always)]
    fn ulps_eq(a: T, b: T, epsilon: u64, max_ulps: u32) -> bool {
        if R::abs_diff_eq(a, b, epsilon) {
            return true;
        }
        R::is_sign_negative(a) == R::is_sign_negative(b) && R::abs_diff_eq(a, b, max_ulps.into())
    }
}

/// The transcendental functions of the quaternion algebra, over a `Real` + `Transcendental`
/// scalar (the split mirrors `UnitQuaternionTrait` / `UnitQuaternionAngleTrait`).
///
/// `fixed` has no hyperbolic functions (escalated to fixed-cairo, WP 8.4-P08 report): `cosh` and
/// `sinh` are composed here from two `exp` and ONE fused kernel each (`(e^z ± e^-z) / 2`,
/// exactly floored). `sinh(z) / z` is then only ever multiplied by a component of `v` (whose
/// magnitude is at most `z = |v|`), so the cancellation of `sinh` near zero costs a few ulp
/// absolute on the result, not a relative error. `exp` panics for arguments above about 21.49
/// (`Fixed: exp overflow`), which bounds the real part (`exp`, `sinh`, `cosh`) or the imaginary
/// norm (`sin`, `cos`) of the accepted inputs.
///
/// The inverse functions (`acos`, `asin`, `atan`, `asinh`, `acosh`, `atanh`) are upstream's
/// compositions of `ln`, `sqrt` and products, in upstream's order; every intermediate rounds,
/// so their error is a few hundred ulp on moderate inputs (oracle tolerances in
/// `tests_ext.cairo`). `acos`, `asin` and `atan` normalise the imaginary part, and `asinh`,
/// `acosh`, `atanh` take the `ln` / `sqrt` of a real quaternion when the argument is real: all of
/// them panic with `errors::REAL_QUATERNION` on a real quaternion, where upstream returns `NaN`
/// (or, for `atan`, panics).
#[generate_trait]
pub impl QuaternionTranscendentalImpl<
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
> of QuaternionTranscendentalTrait<T> {
    // --- polar form ----------------------------------------------------------------------------

    /// `(|self|, θ, axis)` with `self = |self| · (cos θ, axis · sin θ)`: the norm, the HALF
    /// angle `θ = atan2(|v|, |w|)` in `[0, π/2]` (upstream's `UnitQuaternion::angle / 2`, whose
    /// `|w|` makes it ignore the sign of the real part) and the direction of the imaginary part.
    /// The axis is `None` for a real quaternion (then `θ = 0`), and everything is zero for the
    /// zero quaternion. One `norm4`, one `atan2` (scale invariant: no division of the
    /// quaternion by its norm), one normalisation of `v`. Upstream: `polar_decomposition`.
    fn polar_decomposition(self: Quaternion<T>) -> (T, T, Option<Unit<Vector3<T>>>) {
        let n = R::norm4(self.i, self.j, self.k, self.w);
        if n == R::zero() {
            return (R::zero(), R::zero(), None);
        }
        match UnitTrait::try_new_and_get(Vector3 { x: self.i, y: self.j, z: self.k }, R::zero()) {
            Option::Some((axis, nv)) => (n, Tr::atan2(nv, R::abs(self.w)), Some(axis)),
            Option::None => (n, R::zero(), None),
        }
    }

    /// `scale · (cos θ, axis · sin θ)`, the inverse of `polar_decomposition`. One `sin_cos` of
    /// `θ` (upstream's `from_axis_angle(axis, 2θ)` halves the doubled angle back, exactly), then
    /// each imaginary component is the exact triple product `axis_i · sin θ · scale` floored
    /// once (`Real::wide_mul_scalar`), where upstream rounds twice. `axis` must be a unit vector.
    /// Upstream: `Quaternion::from_polar_decomposition`.
    fn from_polar_decomposition(scale: T, theta: T, axis: Unit<Vector3<T>>) -> Quaternion<T> {
        let (s, c) = Tr::sin_cos(theta);
        let a = axis.value;
        Quaternion {
            i: R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), a.x, s), scale),
            j: R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), a.y, s), scale),
            k: R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), a.z, s), scale),
            w: c * scale,
        }
    }

    // --- exponential, logarithm, powers ---------------------------------------------------------

    /// `exp_eps` with upstream's default threshold, `Real::default_epsilon()` (1 ulp).
    /// Upstream: `exp`.
    #[inline(always)]
    fn exp(self: Quaternion<T>) -> Quaternion<T> {
        Self::exp_eps(self, R::default_epsilon())
    }

    /// `e^self = e^w · (cos |v|, v · sin|v| / |v|)`: one `exp`, one `norm3`, one `sin_cos`, one
    /// division for the common factor `sin|v| / |v|`, then each imaginary component is the exact
    /// triple product `v_i · (sin|v| / |v|) · e^w` floored once.
    ///
    /// When `|v| <= eps` the result is the IDENTITY, like upstream, whatever the real part: the
    /// exponential of a real quaternion `(w, 0, 0, 0)` is `1`, not `e^w` (upstream's
    /// `le.if_else(Self::identity, ..)`, kept as is under the owner's "same as the Rust reference"
    /// rule; `e^w` of a real `w` is `Real`'s scalar `exp`). No `exp` is evaluated there, so a huge
    /// real part cannot overflow. `|v|` is compared to `eps` directly: upstream's `|v|² <= eps²`
    /// would floor to zero below `|v| = 2^-16` in fixed point. Panics with `Fixed: exp overflow`
    /// for `w` above about 21.49 otherwise. Upstream: `exp_eps`.
    fn exp_eps(self: Quaternion<T>, eps: T) -> Quaternion<T> {
        let Quaternion { i, j, k, w } = self;
        let n = R::norm3(i, j, k);
        if n <= eps {
            return QuaternionTrait::identity();
        }
        let ew = Tr::exp(w);
        let (s, c) = Tr::sin_cos(n);
        let f = R::div(s, n);
        Quaternion {
            i: R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), i, f), ew),
            j: R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), j, f), ew),
            k: R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), k, f), ew),
            w: ew * c,
        }
    }

    /// The principal logarithm `(ln |self|, v / |v| · θ)` with `θ = acos(w / |self|)` in `[0,
    /// π]`.
    ///
    /// `θ` is computed as `atan2(|v|, w)` — the same angle, but well conditioned where
    /// `acos(w / n)` is not (near `w = ±n`, i.e. a small imaginary part, `acos` loses half the
    /// bits) and with no division: 73 770 gas against 75 370 for the upstream form
    /// (`bench_quaternion_ln__alt_acos`; `test_ln_alt_acos_loses_precision_near_the_real_axis`
    /// shows the accuracy it loses). The direction is
    /// `v / |v|` (three correctly rounded divisions), then scaled by `θ`, like
    /// `UnitQuaternion::scaled_axis`.
    ///
    /// A REAL quaternion (`v = 0`, zero included) panics with `errors::REAL_QUATERNION`: upstream
    /// normalises the zero imaginary part and returns `NaN` there (PLAN M8 fidelity rules).
    /// Upstream: `ln`.
    fn ln(self: Quaternion<T>) -> Quaternion<T> {
        let Quaternion { i, j, k, w } = self;
        let nv = R::norm3(i, j, k);
        if nv == R::zero() {
            core::panic_with_felt252(errors::REAL_QUATERNION);
        }
        let ln_n = Tr::ln(R::norm4(i, j, k, w));
        let theta = Tr::atan2(nv, w);
        let (x, y, z) = R::div3(i, j, k, nv);
        Quaternion { i: x * theta, j: y * theta, k: z * theta, w: ln_n }
    }

    /// `self^n = exp(n · ln(self))` (the product floored per component). Inherits the domain of
    /// `ln` (panics on a real quaternion, where upstream returns `NaN`) and of `exp`. Upstream:
    /// `powf`.
    #[inline(always)]
    fn powf(self: Quaternion<T>, n: T) -> Quaternion<T> {
        Self::exp(Self::ln(self).scale(n))
    }

    // --- trigonometric functions ---------------------------------------------------------------

    /// `cos(self) = (cos w · cosh|v|, -v · sin w · sinh|v| / |v|)`. One `sin_cos`, two `exp`
    /// (`cosh` / `sinh`), one division, then exact triple products floored once per imaginary
    /// component. Panics for `|v|` above about 21.49. Upstream: `cos`.
    fn cos(self: Quaternion<T>) -> Quaternion<T> {
        let (sw, cw) = Tr::sin_cos(self.w);
        let (ch, f) = QuaternionTranscendentalInternalTrait::<T>::cosh_sinhc(self);
        QuaternionTranscendentalInternalTrait::<T>::scale_parts(self, cw * ch, f, -sw)
    }

    /// `sin(self) = (sin w · cosh|v|, v · cos w · sinh|v| / |v|)`, like `cos`. Upstream: `sin`.
    fn sin(self: Quaternion<T>) -> Quaternion<T> {
        let (sw, cw) = Tr::sin_cos(self.w);
        let (ch, f) = QuaternionTranscendentalInternalTrait::<T>::cosh_sinhc(self);
        QuaternionTranscendentalInternalTrait::<T>::scale_parts(self, sw * ch, f, cw)
    }

    /// `sin(self) · cos(self)⁻¹` (`right_div`). Panics with `nalgebra: not invertible` when
    /// `|cos(self)|²` floors to zero (upstream unwraps `None`). Upstream: `tan`.
    fn tan(self: Quaternion<T>) -> Quaternion<T> {
        Self::sin(self).right_div(Self::cos(self)).expect(errors::NOT_INVERTIBLE)
    }

    /// `-(u · ln(self + sqrt(self² - 1)))` with `u = (0, v / |v|)`, upstream's composition.
    /// Panics with `errors::REAL_QUATERNION` on a real quaternion (upstream: `NaN`). Upstream:
    /// `acos`.
    fn acos(self: Quaternion<T>) -> Quaternion<T> {
        let u = QuaternionTranscendentalInternalTrait::<T>::unit_imag(self);
        let z = Self::ln(self + (self.squared() - QuaternionTrait::identity()).sqrt());
        -(u * z)
    }

    /// `-(u · ln(u · self + sqrt(1 - self²)))` with `u = (0, v / |v|)`, upstream's composition.
    /// Panics with `errors::REAL_QUATERNION` on a real quaternion (upstream: `NaN`). Upstream:
    /// `asin`.
    fn asin(self: Quaternion<T>) -> Quaternion<T> {
        let u = QuaternionTranscendentalInternalTrait::<T>::unit_imag(self);
        let z = Self::ln(u * self + (QuaternionTrait::identity() - self.squared()).sqrt());
        -(u * z)
    }

    /// `(u / 2) · ln((u + self) · (u - self)⁻¹)` with `u = (0, v / |v|)`, upstream's
    /// composition. Panics with `errors::REAL_QUATERNION` on a real quaternion (where upstream
    /// panics too) and with `nalgebra: not invertible` when `u - self` is not invertible.
    /// Upstream: `atan`.
    fn atan(self: Quaternion<T>) -> Quaternion<T> {
        let u = QuaternionTranscendentalInternalTrait::<T>::unit_imag(self);
        let fr = (u + self).right_div(u - self).expect(errors::NOT_INVERTIBLE);
        u.half() * Self::ln(fr)
    }

    // --- hyperbolic functions ------------------------------------------------------------------

    /// `sinh(self) = (sinh w · cos|v|, v · cosh w · sin|v| / |v|)`: the closed form of
    /// upstream's `(exp(self) - exp(-self)) / 2`, sharing one `sin_cos` and two scalar `exp`
    /// between the two exponentials: 94 150 gas against 157 140 for upstream's composition of
    /// two full quaternion `exp` (`bench_quaternion_sinh__alt_exp_difference`, agreeing to the
    /// oracle tolerance: `test_sinh_alt_exp_difference_agrees`). When `|v| <= default_epsilon`
    /// both exponentials are the identity upstream (see `exp_eps`), so `sinh` is ZERO there, like
    /// upstream, whatever the real part. Panics for `|w|` above about 21.49 otherwise. Upstream:
    /// `sinh`.
    fn sinh(self: Quaternion<T>) -> Quaternion<T> {
        match QuaternionTranscendentalInternalTrait::<T>::sinc_cos(self) {
            Some((
                f, c,
            )) => {
                let (ch, sh) = QuaternionTranscendentalInternalTrait::<T>::cosh_sinh(self.w);
                QuaternionTranscendentalInternalTrait::<T>::scale_parts(self, sh * c, f, ch)
            },
            None => Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::zero() },
        }
    }

    /// `cosh(self) = (cosh w · cos|v|, v · sinh w · sin|v| / |v|)`, the closed form of
    /// upstream's `(exp(self) + exp(-self)) / 2` (see `sinh`): the IDENTITY when `|v| <=
    /// default_epsilon`, like upstream, whatever the real part. Upstream: `cosh`.
    fn cosh(self: Quaternion<T>) -> Quaternion<T> {
        match QuaternionTranscendentalInternalTrait::<T>::sinc_cos(self) {
            Some((
                f, c,
            )) => {
                let (ch, sh) = QuaternionTranscendentalInternalTrait::<T>::cosh_sinh(self.w);
                QuaternionTranscendentalInternalTrait::<T>::scale_parts(self, ch * c, f, sh)
            },
            None => QuaternionTrait::identity(),
        }
    }

    /// `sinh(self) · cosh(self)⁻¹` (`right_div`): zero when `|v| <= default_epsilon`, like
    /// upstream (`sinh` is zero and `cosh` the identity there). Panics with
    /// `nalgebra: not invertible` when `|cosh(self)|²` floors to zero. Upstream: `tanh`.
    fn tanh(self: Quaternion<T>) -> Quaternion<T> {
        Self::sinh(self).right_div(Self::cosh(self)).expect(errors::NOT_INVERTIBLE)
    }

    /// `ln(self + sqrt(1 + self²))`, upstream's composition. Upstream: `asinh`.
    fn asinh(self: Quaternion<T>) -> Quaternion<T> {
        Self::ln(self + (QuaternionTrait::identity() + self.squared()).sqrt())
    }

    /// `ln(self + sqrt(self + 1) · sqrt(self - 1))`, upstream's composition. Upstream: `acosh`.
    fn acosh(self: Quaternion<T>) -> Quaternion<T> {
        let one = QuaternionTrait::identity();
        Self::ln(self + (self + one).sqrt() * (self - one).sqrt())
    }

    /// `(ln(1 + self) - ln(1 - self)) / 2`, upstream's composition. Panics with
    /// `errors::REAL_QUATERNION` on a real quaternion (upstream: `NaN`). Upstream: `atanh`.
    fn atanh(self: Quaternion<T>) -> Quaternion<T> {
        let one = QuaternionTrait::identity();
        (Self::ln(one + self) - Self::ln(one - self)).half()
    }
}

/// Crate-internal kernels of `QuaternionTranscendentalTrait`.
#[generate_trait]
pub(crate) impl QuaternionTranscendentalInternalImpl<
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
> of QuaternionTranscendentalInternalTrait<T> {
    /// `(cosh x, sinh x)` from `e^x` and `e^-x`, each `(e^x ± e^-x) / 2` one fused kernel
    /// (exactly floored). `fixed` has no hyperbolic functions (escalation, see the trait doc).
    #[inline(always)]
    fn cosh_sinh(x: T) -> (T, T) {
        let (ep, em) = (Tr::exp(x), Tr::exp(-x));
        let ch = R::wide_rescale(
            R::wide_add_prod(R::wide_add_prod(R::wide_zero(), ep, R::HALF), em, R::HALF),
        );
        let sh = R::wide_rescale(
            R::wide_sub_prod(R::wide_add_prod(R::wide_zero(), ep, R::HALF), em, R::HALF),
        );
        (ch, sh)
    }

    /// `(cosh z, sinh z / z)` with `z = |v|` (and `(1, 1)` for `v = 0`, where the factor is
    /// irrelevant: it multiplies a zero `v`).
    fn cosh_sinhc(q: Quaternion<T>) -> (T, T) {
        let z = R::norm3(q.i, q.j, q.k);
        if z == R::zero() {
            return (R::one(), R::one());
        }
        let (ch, sh) = Self::cosh_sinh(z);
        (ch, R::div(sh, z))
    }

    /// `(sin z / z, cos z)` with `z = |v|`, or `None` when `z <= default_epsilon`: the threshold
    /// under which upstream's `exp` returns the identity (`exp_eps`), hence `sinh` / `cosh`
    /// (compositions of two `exp` upstream) their `|v| = 0` values.
    fn sinc_cos(q: Quaternion<T>) -> Option<(T, T)> {
        let z = R::norm3(q.i, q.j, q.k);
        if z <= R::default_epsilon() {
            return None;
        }
        let (s, c) = Tr::sin_cos(z);
        Some((R::div(s, z), c))
    }

    /// `(w, v · f · g)`: each imaginary component the exact triple product `v_i · f · g`,
    /// floored once.
    #[inline(always)]
    fn scale_parts(q: Quaternion<T>, w: T, f: T, g: T) -> Quaternion<T> {
        Quaternion {
            i: R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), q.i, f), g),
            j: R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), q.j, f), g),
            k: R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), q.k, f), g),
            w,
        }
    }

    /// `(0, v / |v|)`. Panics with `errors::REAL_QUATERNION` for `v = 0` (upstream: `NaN`).
    #[inline(always)]
    fn unit_imag(q: Quaternion<T>) -> Quaternion<T> {
        let nv = R::norm3(q.i, q.j, q.k);
        if nv == R::zero() {
            core::panic_with_felt252(errors::REAL_QUATERNION);
        }
        let (i, j, k) = R::div3(q.i, q.j, q.k, nv);
        Quaternion { i, j, k, w: R::zero() }
    }
}

/// `a + b`, component-wise. Exact; panics on overflow. Upstream: `Add`.
pub impl QuaternionAdd<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<Quaternion<T>> {
    #[inline(always)]
    fn add(lhs: Quaternion<T>, rhs: Quaternion<T>) -> Quaternion<T> {
        Quaternion { i: lhs.i + rhs.i, j: lhs.j + rhs.j, k: lhs.k + rhs.k, w: lhs.w + rhs.w }
    }
}

/// `a - b`, component-wise. Exact; panics on overflow. Upstream: `Sub`.
pub impl QuaternionSub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<Quaternion<T>> {
    #[inline(always)]
    fn sub(lhs: Quaternion<T>, rhs: Quaternion<T>) -> Quaternion<T> {
        Quaternion { i: lhs.i - rhs.i, j: lhs.j - rhs.j, k: lhs.k - rhs.k, w: lhs.w - rhs.w }
    }
}

/// `-a`, component-wise: the same rotation when `a` is a unit quaternion. Exact; panics on
/// overflow (`-MIN`). Upstream: `Neg`.
pub impl QuaternionNeg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Quaternion<T>> {
    #[inline(always)]
    fn neg(a: Quaternion<T>) -> Quaternion<T> {
        Quaternion { i: -a.i, j: -a.j, k: -a.k, w: -a.w }
    }
}

/// The Hamilton product `lhs * rhs` (`i·j = k`, non-commutative): the composition of the two
/// rotations when both are unit quaternions, `lhs` applied last.
///
/// Each of the four components is ONE fused kernel of four products, with the signs folded into the
/// accumulation (`Real::wide_*`): the exact sum of the four products is floored once and checked
/// once, so the result is the exactly floored product, bit for bit, and intermediate products may
/// exceed the scalar range. 16 products, 4 roundings, 11 860 gas.
///
/// Carrying the three minus signs in negated operands of `lhs` and using `Real::sum_prod4` instead
/// gives the same bits but costs 12 760 (the three negations), and panics on a component equal to
/// the scalar's `MIN` (`bench_quaternion_mul__alt_sum_prod4`, `test_mul_alt_sum_prod4_*`). An
/// unfused product (one rounding per product, forbidden by AGENTS.md rule 4) costs 38 780 and
/// overflows on intermediate products. Panics on overflow of a component. Upstream: `Mul`
/// (`quaternion_ops.rs`).
pub impl QuaternionMul<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of Mul<Quaternion<T>> {
    fn mul(lhs: Quaternion<T>, rhs: Quaternion<T>) -> Quaternion<T> {
        // w = aw·bw - ai·bi - aj·bj - ak·bk
        let w = R::wide_add_prod(R::wide_zero(), lhs.w, rhs.w);
        let w = R::wide_sub_prod(R::wide_sub_prod(w, lhs.i, rhs.i), lhs.j, rhs.j);
        let w = R::wide_rescale(R::wide_sub_prod(w, lhs.k, rhs.k));
        // i = aw·bi + ai·bw + aj·bk - ak·bj
        let i = R::wide_add_prod(R::wide_zero(), lhs.w, rhs.i);
        let i = R::wide_add_prod(R::wide_add_prod(i, lhs.i, rhs.w), lhs.j, rhs.k);
        let i = R::wide_rescale(R::wide_sub_prod(i, lhs.k, rhs.j));
        // j = aw·bj - ai·bk + aj·bw + ak·bi
        let j = R::wide_add_prod(R::wide_zero(), lhs.w, rhs.j);
        let j = R::wide_add_prod(R::wide_sub_prod(j, lhs.i, rhs.k), lhs.j, rhs.w);
        let j = R::wide_rescale(R::wide_add_prod(j, lhs.k, rhs.i));
        // k = aw·bk + ai·bj - aj·bi + ak·bw
        let k = R::wide_add_prod(R::wide_zero(), lhs.w, rhs.k);
        let k = R::wide_sub_prod(R::wide_add_prod(k, lhs.i, rhs.j), lhs.j, rhs.i);
        let k = R::wide_rescale(R::wide_add_prod(k, lhs.k, rhs.w));
        Quaternion { i, j, k, w }
    }
}

/// `Zero::zero()`: the quaternion `0`; `is_zero` / `is_non_zero` compare the four components with
/// zero exactly. Upstream: `num::Zero for Quaternion` (the former `QuaternionTrait::zero`, WP 8.0).
pub impl QuaternionZero<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Zero<Quaternion<T>> {
    #[inline(always)]
    fn zero() -> Quaternion<T> {
        Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::zero() }
    }

    #[inline(always)]
    fn is_zero(self: @Quaternion<T>) -> bool {
        *self.i == R::zero() && *self.j == R::zero() && *self.k == R::zero() && *self.w == R::zero()
    }

    #[inline(always)]
    fn is_non_zero(self: @Quaternion<T>) -> bool {
        !Self::is_zero(self)
    }
}

/// `coords.into()`: the quaternion of coordinates `(i, j, k, w)` (the STORAGE order). Upstream:
/// `From<Vector4> for Quaternion`.
pub impl QuaternionFromVector<T> of Into<Vector4<T>, Quaternion<T>> {
    #[inline(always)]
    fn into(self: Vector4<T>) -> Quaternion<T> {
        let Vector4 { x, y, z, w } = self;
        Quaternion { i: x, j: y, k: z, w }
    }
}

/// `One::one()`: the identity quaternion `1`; `is_one` / `is_non_one` compare the four components
/// with `(1, 0, 0, 0)` exactly. Upstream: `num::One for Quaternion`.
pub impl QuaternionOne<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of One<Quaternion<T>> {
    #[inline(always)]
    fn one() -> Quaternion<T> {
        Quaternion { i: R::zero(), j: R::zero(), k: R::zero(), w: R::one() }
    }

    #[inline(always)]
    fn is_one(self: @Quaternion<T>) -> bool {
        *self.i == R::zero() && *self.j == R::zero() && *self.k == R::zero() && *self.w == R::one()
    }

    #[inline(always)]
    fn is_non_one(self: @Quaternion<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `q[i]`: the coordinate `i` in the STORAGE order `(i, j, k, w)` (`q[3]` is the real part), like
/// upstream's `Index<usize>` on `coords`. Panics with `nalgebra: index out of bounds` for
/// `index > 3`. Upstream: `Index<usize> for Quaternion`.
pub impl QuaternionIndex<T, +Copy<T>, +Drop<T>> of Index<Quaternion<T>, usize> {
    type Target = T;

    #[inline(always)]
    fn index(ref self: Quaternion<T>, index: usize) -> T {
        match index {
            0 => self.i,
            1 => self.j,
            2 => self.k,
            3 => self.w,
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}

/// `[i, j, k, w].into()`: the quaternion of coordinates `(i, j, k, w)` (the STORAGE order, not
/// `new`'s). Upstream: `From<[T; 4]> for Quaternion`.
pub impl QuaternionFromArray<T, +Drop<T>> of Into<[T; 4], Quaternion<T>> {
    #[inline(always)]
    fn into(self: [T; 4]) -> Quaternion<T> {
        let [i, j, k, w] = self;
        Quaternion { i, j, k, w }
    }
}
