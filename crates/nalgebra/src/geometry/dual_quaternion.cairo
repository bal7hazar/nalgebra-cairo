//! `DualQuaternion`: a dual quaternion `real + ε·dual` (upstream `nalgebra::DualQuaternion`,
//! WP 8.4-P12).
//!
//! - `DualQuaternionTrait` / `DualQuaternionImpl`: construction, conjugation, normalisation,
//!   inverse, interpolation, the scalar products (`scale` / `unscale`, upstream's `* k` / `/ k`),
//!   the approximate comparisons and the products by a `UnitDualQuaternion`
//!   (`mul_unit_dual_quaternion` / `div_unit_dual_quaternion`: Cairo's `Mul` / `Div` are
//!   homogeneous);
//! - the operators `+`, `-`, unary `-`, `*` (the dual-quaternion product), `*=` / `/=` by a scalar,
//!   `Zero`, `One` and `Index`: their impls live in this module, where the compiler finds them
//!   without any import.
//!
//! The unit dual quaternion of a rigid-body transform is `UnitDualQuaternion`
//! (`geometry::unit_dual_quaternion`); this type is the general algebra it is built on.
//!
//! Conventions: the fields are upstream's (`real`, `dual`), and `Serde` / `Index` follow
//! upstream's `[T; 8]` view `(real.i, real.j, real.k, real.w, dual.i, dual.j, dual.k, dual.w)`
//! (the storage order of each `Quaternion`, DESIGN D8).
//!
//! The approximate comparisons (`abs_diff_eq`, `relative_eq`, `ulps_eq`) count their tolerances
//! in ulp (DESIGN D3) and, like upstream's `approx` impls, accept `other` OR `-other` taken as a
//! WHOLE (the eight components compared with the same sign: `(r, d)` and `(-r, -d)` are the same
//! rigid transform, `(r, -d)` is not).
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently. In
//! particular each component of the dual part of a product, `a.real · b.dual + a.dual · b.real`,
//! is ONE accumulation of eight products floored once, where upstream rounds two Hamilton
//! products and their sum.

use core::num::traits::{One, Zero};
use core::ops::{DivAssign, Index, MulAssign};
use simba::scalar::Real;
use super::quaternion::{ApproxEqTrait, Quaternion, QuaternionTrait};
use super::unit_dual_quaternion::{UnitDualQuaternion, UnitDualQuaternionTrait};

/// Panic messages of the dual-quaternion algebra (stable API).
pub mod errors {
    /// `dq[i]` with `i > 7`.
    pub const INDEX_OUT_OF_BOUNDS: felt252 = 'nalgebra: index out of bounds';
}

/// A dual quaternion `real + ε·dual` (`ε² = 0`). When it is a unit dual quaternion (see
/// `UnitDualQuaternion`), `real` is the rotation and `dual = t · real / 2` encodes the
/// translation `t`. Upstream: `DualQuaternion<T>` (public fields `real`, `dual`).
///
/// `Default` is the ZERO dual quaternion, like upstream's `Default` (the identity is
/// `DualQuaternionTrait::identity` / `One::one`).
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug)]
pub struct DualQuaternion<T> {
    pub real: Quaternion<T>,
    pub dual: Quaternion<T>,
}

/// Methods of `DualQuaternion<T>` for any `Real` scalar. By value, unrolled, no loop.
#[generate_trait]
pub impl DualQuaternionImpl<
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
> of DualQuaternionTrait<T> {
    // --- construction --------------------------------------------------------------------------

    /// The dual quaternion `real + ε·dual`. Exact. Upstream:
    /// `DualQuaternion::from_real_and_dual`.
    #[inline(always)]
    fn from_real_and_dual(real: Quaternion<T>, dual: Quaternion<T>) -> DualQuaternion<T> {
        DualQuaternion { real, dual }
    }

    /// The dual quaternion `real + ε·0`. Exact. Upstream: `DualQuaternion::from_real`.
    #[inline(always)]
    fn from_real(real: Quaternion<T>) -> DualQuaternion<T> {
        DualQuaternion { real, dual: Zero::zero() }
    }

    /// The multiplicative identity `1 + ε·0`. Exact. Upstream: `DualQuaternion::identity`.
    #[inline(always)]
    fn identity() -> DualQuaternion<T> {
        DualQuaternion { real: QuaternionTrait::identity(), dual: Zero::zero() }
    }

    /// The same dual quaternion with every component converted by `Into<T, U>` (the identity for
    /// the single scalar `Fixed`). Upstream: `DualQuaternion::cast` (and
    /// `SubsetOf<DualQuaternion>`, the `nalgebra::convert` it goes through).
    fn cast<U, +Into<T, U>, +Drop<U>>(self: DualQuaternion<T>) -> DualQuaternion<U> {
        DualQuaternion { real: self.real.cast(), dual: self.dual.cast() }
    }

    // --- algebra -------------------------------------------------------------------------------

    /// `(real.conjugate(), dual.conjugate())`, the quaternion conjugate of both parts. Exact;
    /// panics on overflow (`-MIN`). Upstream: `conjugate`.
    #[inline(always)]
    fn conjugate(self: DualQuaternion<T>) -> DualQuaternion<T> {
        DualQuaternion { real: self.real.conjugate(), dual: self.dual.conjugate() }
    }

    /// `conjugate`, in place. Upstream: `conjugate_mut`.
    #[inline(always)]
    fn conjugate_mut(ref self: DualQuaternion<T>) {
        self = DualQuaternion { real: self.real.conjugate(), dual: self.dual.conjugate() };
    }

    /// `self * k`, each of the eight components floored once. Panics on overflow. Upstream:
    /// `Mul<T>` (`dq * k`, and `k * dq`).
    #[inline(always)]
    fn scale(self: DualQuaternion<T>, k: T) -> DualQuaternion<T> {
        DualQuaternion { real: self.real.scale(k), dual: self.dual.scale(k) }
    }

    /// `self / k`, each component the correctly rounded quotient (one division per component, see
    /// `QuaternionTrait::unscale`). Panics on a zero `k` and on overflow. Upstream: `Div<T>`
    /// (`dq / k`).
    #[inline(always)]
    fn unscale(self: DualQuaternion<T>, k: T) -> DualQuaternion<T> {
        DualQuaternion { real: self.real.unscale(k), dual: self.dual.unscale(k) }
    }

    /// Both parts divided by the norm of the REAL part (`|real|`, `Real::norm4`, floored once):
    /// one correctly rounded division per component. The result is a unit dual quaternion when
    /// `self` encodes a rigid transform (`real · dual* + dual · real* = 0`, which a common factor
    /// preserves). Panics with `Fixed: division by zero` on a zero real part. Upstream:
    /// `normalize`.
    #[inline(always)]
    fn normalize(self: DualQuaternion<T>) -> DualQuaternion<T> {
        let n = self.real.norm();
        DualQuaternion { real: self.real.unscale(n), dual: self.dual.unscale(n) }
    }

    /// `normalize`, in place; returns the norm of the real part it divided by. Upstream:
    /// `normalize_mut`.
    #[inline(always)]
    fn normalize_mut(ref self: DualQuaternion<T>) -> T {
        let n = self.real.norm();
        self = DualQuaternion { real: self.real.unscale(n), dual: self.dual.unscale(n) };
        n
    }

    /// The multiplicative inverse `(r⁻¹, -r⁻¹ · dual · r⁻¹)` with `r⁻¹ =
    /// real.try_inverse()`, or `None` when the real part is not invertible (`|real|²` floors to
    /// zero, see `QuaternionTrait::try_inverse`). The dual part is upstream's `(-r⁻¹ · dual) ·
    /// r⁻¹`, in upstream's order: the negation is exact, then two fused Hamilton products (each
    /// component floored once per product). Panics on overflow. Upstream: `try_inverse`.
    fn try_inverse(self: DualQuaternion<T>) -> Option<DualQuaternion<T>> {
        match self.real.try_inverse() {
            Some(inv) => Some(DualQuaternion { real: inv, dual: (-inv * self.dual) * inv }),
            None => None,
        }
    }

    /// `try_inverse`, in place: returns `false` and leaves `self` unchanged when the real part is
    /// not invertible. Upstream: `try_inverse_mut` (which also leaves `self` unchanged then).
    fn try_inverse_mut(ref self: DualQuaternion<T>) -> bool {
        match Self::try_inverse(self) {
            Some(inv) => {
                self = inv;
                true
            },
            None => false,
        }
    }

    /// The linear interpolation of both parts, `QuaternionTrait::lerp` per part (`self + (other -
    /// self) · t` per component, one floor rounding each). `t` is not clamped; `t = 0` gives
    /// `self` and `t = 1` gives `other` exactly. Panics on overflow. Upstream: `lerp`
    /// (`self * (1 - t) + other * t`).
    #[inline(always)]
    fn lerp(self: DualQuaternion<T>, other: DualQuaternion<T>, t: T) -> DualQuaternion<T> {
        DualQuaternion { real: self.real.lerp(other.real, t), dual: self.dual.lerp(other.dual, t) }
    }

    // --- approximate comparisons (tolerances in ulp, DESIGN D3) --------------------------------

    /// `true` when all eight components are within `ulps` smallest units (raw units for fixed
    /// point) of `other`'s, or all eight within `ulps` of `-other`'s (upstream compares the
    /// `[T; 8]` views, then their negation as a whole). The second comparison runs only when the
    /// first fails; it negates `other`, hence panics on a component equal to the scalar's `MIN`
    /// there. Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp
    /// instead of a float epsilon.
    fn abs_diff_eq(self: DualQuaternion<T>, other: DualQuaternion<T>, ulps: u64) -> bool {
        let (a, b, c, d) = (self.real, self.dual, other.real, other.dual);
        (R::abs_diff_eq(a.i, c.i, ulps)
            && R::abs_diff_eq(a.j, c.j, ulps)
            && R::abs_diff_eq(a.k, c.k, ulps)
            && R::abs_diff_eq(a.w, c.w, ulps)
            && R::abs_diff_eq(b.i, d.i, ulps)
            && R::abs_diff_eq(b.j, d.j, ulps)
            && R::abs_diff_eq(b.k, d.k, ulps)
            && R::abs_diff_eq(b.w, d.w, ulps))
            || (R::abs_diff_eq(a.i, -c.i, ulps)
                && R::abs_diff_eq(a.j, -c.j, ulps)
                && R::abs_diff_eq(a.k, -c.k, ulps)
                && R::abs_diff_eq(a.w, -c.w, ulps)
                && R::abs_diff_eq(b.i, -d.i, ulps)
                && R::abs_diff_eq(b.j, -d.j, ulps)
                && R::abs_diff_eq(b.k, -d.k, ulps)
                && R::abs_diff_eq(b.w, -d.w, ulps))
    }

    /// `QuaternionTrait::relative_eq`'s per-component test on all eight components, against
    /// `other` or against `-other` as a whole (see `abs_diff_eq`). Upstream:
    /// `approx::RelativeEq::relative_eq`, `epsilon` counted in ulp.
    fn relative_eq(
        self: DualQuaternion<T>, other: DualQuaternion<T>, epsilon: u64, max_relative: T,
    ) -> bool {
        let (a, b, c, d) = (self.real, self.dual, other.real, other.dual);
        (ApproxEqTrait::relative_eq(a.i, c.i, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.j, c.j, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.k, c.k, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(a.w, c.w, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(b.i, d.i, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(b.j, d.j, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(b.k, d.k, epsilon, max_relative)
            && ApproxEqTrait::relative_eq(b.w, d.w, epsilon, max_relative))
            || (ApproxEqTrait::relative_eq(a.i, -c.i, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(a.j, -c.j, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(a.k, -c.k, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(a.w, -c.w, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(b.i, -d.i, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(b.j, -d.j, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(b.k, -d.k, epsilon, max_relative)
                && ApproxEqTrait::relative_eq(b.w, -d.w, epsilon, max_relative))
    }

    /// `QuaternionTrait::ulps_eq`'s per-component test on all eight components, against `other`
    /// or against `-other` as a whole (see `abs_diff_eq`). Upstream: `approx::UlpsEq::ulps_eq`.
    fn ulps_eq(
        self: DualQuaternion<T>, other: DualQuaternion<T>, epsilon: u64, max_ulps: u32,
    ) -> bool {
        let (a, b, c, d) = (self.real, self.dual, other.real, other.dual);
        (ApproxEqTrait::ulps_eq(a.i, c.i, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.j, c.j, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.k, c.k, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(a.w, c.w, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(b.i, d.i, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(b.j, d.j, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(b.k, d.k, epsilon, max_ulps)
            && ApproxEqTrait::ulps_eq(b.w, d.w, epsilon, max_ulps))
            || (ApproxEqTrait::ulps_eq(a.i, -c.i, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(a.j, -c.j, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(a.k, -c.k, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(a.w, -c.w, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(b.i, -d.i, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(b.j, -d.j, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(b.k, -d.k, epsilon, max_ulps)
                && ApproxEqTrait::ulps_eq(b.w, -d.w, epsilon, max_ulps))
    }

    // --- heterogeneous operators (Cairo's `Mul` / `Div` are homogeneous) -----------------------

    /// `self * rhs` with a unit dual quaternion: the dual-quaternion product by
    /// `rhs.dual_quaternion()`, a general dual quaternion. Upstream: `Mul<UnitDualQuaternion> for
    /// DualQuaternion`.
    #[inline(always)]
    fn mul_unit_dual_quaternion(
        self: DualQuaternion<T>, rhs: UnitDualQuaternion<T>,
    ) -> DualQuaternion<T> {
        self * rhs.dual_quaternion
    }

    /// `self / rhs = self * rhs.inverse()` (`UnitDualQuaternionTrait::inverse`, then the fused
    /// product). Upstream: `Div<UnitDualQuaternion> for DualQuaternion`.
    #[inline(always)]
    fn div_unit_dual_quaternion(
        self: DualQuaternion<T>, rhs: UnitDualQuaternion<T>,
    ) -> DualQuaternion<T> {
        self * rhs.inverse().dual_quaternion
    }
}

/// Crate-internal kernels of the dual-quaternion product (shared with `UnitDualQuaternion`):
/// one component of a Hamilton product ADDED to an exact wide accumulator, so that sums of
/// several Hamilton products (`a.real · b.dual + a.dual · b.real`) are floored once per
/// component. Each helper is the term sequence of `QuaternionMul` for its component.
#[generate_trait]
pub(crate) impl DualQuaternionInternalImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of DualQuaternionInternalTrait<T> {
    /// `acc + (a · b).w = acc + aw·bw - ai·bi - aj·bj - ak·bk`, exact.
    #[inline(always)]
    fn acc_w(acc: R::Wide, a: Quaternion<T>, b: Quaternion<T>) -> R::Wide {
        let acc = R::wide_sub_prod(R::wide_add_prod(acc, a.w, b.w), a.i, b.i);
        R::wide_sub_prod(R::wide_sub_prod(acc, a.j, b.j), a.k, b.k)
    }

    /// `acc + (a · b).i = acc + aw·bi + ai·bw + aj·bk - ak·bj`, exact.
    #[inline(always)]
    fn acc_i(acc: R::Wide, a: Quaternion<T>, b: Quaternion<T>) -> R::Wide {
        let acc = R::wide_add_prod(R::wide_add_prod(acc, a.w, b.i), a.i, b.w);
        R::wide_sub_prod(R::wide_add_prod(acc, a.j, b.k), a.k, b.j)
    }

    /// `acc + (a · b).j = acc + aw·bj - ai·bk + aj·bw + ak·bi`, exact.
    #[inline(always)]
    fn acc_j(acc: R::Wide, a: Quaternion<T>, b: Quaternion<T>) -> R::Wide {
        let acc = R::wide_sub_prod(R::wide_add_prod(acc, a.w, b.j), a.i, b.k);
        R::wide_add_prod(R::wide_add_prod(acc, a.j, b.w), a.k, b.i)
    }

    /// `acc + (a · b).k = acc + aw·bk + ai·bj - aj·bi + ak·bw`, exact.
    #[inline(always)]
    fn acc_k(acc: R::Wide, a: Quaternion<T>, b: Quaternion<T>) -> R::Wide {
        let acc = R::wide_add_prod(R::wide_add_prod(acc, a.w, b.k), a.i, b.j);
        R::wide_add_prod(R::wide_sub_prod(acc, a.j, b.i), a.k, b.w)
    }

    /// `a1 · b1 + a2 · b2` (two Hamilton products summed), each component ONE accumulation of
    /// eight products floored once: the dual part of the dual-quaternion product.
    fn mul_add_mul(
        a1: Quaternion<T>, b1: Quaternion<T>, a2: Quaternion<T>, b2: Quaternion<T>,
    ) -> Quaternion<T> {
        Quaternion {
            i: R::wide_rescale(Self::acc_i(Self::acc_i(R::wide_zero(), a1, b1), a2, b2)),
            j: R::wide_rescale(Self::acc_j(Self::acc_j(R::wide_zero(), a1, b1), a2, b2)),
            k: R::wide_rescale(Self::acc_k(Self::acc_k(R::wide_zero(), a1, b1), a2, b2)),
            w: R::wide_rescale(Self::acc_w(Self::acc_w(R::wide_zero(), a1, b1), a2, b2)),
        }
    }
}

/// `a + b`, both parts component-wise. Exact; panics on overflow. Upstream: `Add`.
pub impl DualQuaternionAdd<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<DualQuaternion<T>> {
    #[inline(always)]
    fn add(lhs: DualQuaternion<T>, rhs: DualQuaternion<T>) -> DualQuaternion<T> {
        DualQuaternion { real: lhs.real + rhs.real, dual: lhs.dual + rhs.dual }
    }
}

/// `a - b`, both parts component-wise. Exact; panics on overflow. Upstream: `Sub`.
pub impl DualQuaternionSub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<DualQuaternion<T>> {
    #[inline(always)]
    fn sub(lhs: DualQuaternion<T>, rhs: DualQuaternion<T>) -> DualQuaternion<T> {
        DualQuaternion { real: lhs.real - rhs.real, dual: lhs.dual - rhs.dual }
    }
}

/// `-a`, both parts: the same rigid transform when `a` is a unit dual quaternion. Exact; panics on
/// overflow (`-MIN`). Upstream: `Neg`.
pub impl DualQuaternionNeg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<DualQuaternion<T>> {
    #[inline(always)]
    fn neg(a: DualQuaternion<T>) -> DualQuaternion<T> {
        DualQuaternion { real: -a.real, dual: -a.dual }
    }
}

/// The dual-quaternion product `(a.real · b.real, a.real · b.dual + a.dual · b.real)`: the
/// composition of the two rigid transforms when both are unit, `lhs` applied last.
///
/// The real part is the fused Hamilton product (`QuaternionMul`, 16 products, 4 roundings); each
/// component of the dual part is ONE accumulation of the eight products of the two Hamilton
/// products, floored once — the exact floor of the true result, where upstream rounds each
/// product and the sum (48 products, 8 roundings in all). Panics on overflow of a component.
/// Upstream: `Mul` (`dual_quaternion_ops.rs`).
pub impl DualQuaternionMul<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of Mul<DualQuaternion<T>> {
    fn mul(lhs: DualQuaternion<T>, rhs: DualQuaternion<T>) -> DualQuaternion<T> {
        DualQuaternion {
            real: lhs.real * rhs.real,
            dual: DualQuaternionInternalTrait::mul_add_mul(lhs.real, rhs.dual, lhs.dual, rhs.real),
        }
    }
}

/// `dq *= k`: `dq = dq.scale(k)`. Upstream: `MulAssign<T> for DualQuaternion`.
pub impl DualQuaternionMulAssign<
    T, +Mul<T>, +Copy<T>, +Drop<T>,
> of MulAssign<DualQuaternion<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: DualQuaternion<T>, rhs: T) {
        let (r, d) = (self.real, self.dual);
        self =
            DualQuaternion {
                real: Quaternion { i: r.i * rhs, j: r.j * rhs, k: r.k * rhs, w: r.w * rhs },
                dual: Quaternion { i: d.i * rhs, j: d.j * rhs, k: d.k * rhs, w: d.w * rhs },
            };
    }
}

/// `dq /= k`: `dq = dq.unscale(k)` (one correctly rounded division per component). Upstream:
/// `DivAssign<T> for DualQuaternion`.
pub impl DualQuaternionDivAssign<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of DivAssign<DualQuaternion<T>, T> {
    #[inline(always)]
    fn div_assign(ref self: DualQuaternion<T>, rhs: T) {
        let (r, d) = (self.real, self.dual);
        let (ri, rj, rk, rw) = R::div4(r.i, r.j, r.k, r.w, rhs);
        let (di, dj, dk, dw) = R::div4(d.i, d.j, d.k, d.w, rhs);
        self =
            DualQuaternion {
                real: Quaternion { i: ri, j: rj, k: rk, w: rw },
                dual: Quaternion { i: di, j: dj, k: dk, w: dw },
            };
    }
}

/// `Zero::zero()`: the dual quaternion `0 + ε·0`; `is_zero` compares the eight components with
/// zero exactly. Upstream: `num::Zero for DualQuaternion`.
pub impl DualQuaternionZero<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Zero<DualQuaternion<T>> {
    #[inline(always)]
    fn zero() -> DualQuaternion<T> {
        DualQuaternion { real: Zero::zero(), dual: Zero::zero() }
    }

    #[inline(always)]
    fn is_zero(self: @DualQuaternion<T>) -> bool {
        self.real.is_zero() && self.dual.is_zero()
    }

    #[inline(always)]
    fn is_non_zero(self: @DualQuaternion<T>) -> bool {
        !Self::is_zero(self)
    }
}

/// `One::one()`: the identity `1 + ε·0`; `is_one` compares the eight components exactly.
/// Upstream: `num::One for DualQuaternion`.
pub impl DualQuaternionOne<
    T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of One<DualQuaternion<T>> {
    #[inline(always)]
    fn one() -> DualQuaternion<T> {
        DualQuaternion { real: One::one(), dual: Zero::zero() }
    }

    #[inline(always)]
    fn is_one(self: @DualQuaternion<T>) -> bool {
        self.real.is_one() && self.dual.is_zero()
    }

    #[inline(always)]
    fn is_non_one(self: @DualQuaternion<T>) -> bool {
        !Self::is_one(self)
    }
}

/// `dq[i]`: the component `i` of upstream's `[T; 8]` view `(real.i, real.j, real.k, real.w,
/// dual.i, dual.j, dual.k, dual.w)`. Panics with `nalgebra: index out of bounds` for `i > 7`.
/// Upstream: `Index<usize> for DualQuaternion`.
pub impl DualQuaternionIndex<T, +Copy<T>, +Drop<T>> of Index<DualQuaternion<T>, usize> {
    type Target = T;

    #[inline(always)]
    fn index(ref self: DualQuaternion<T>, index: usize) -> T {
        match index {
            0 => self.real.i,
            1 => self.real.j,
            2 => self.real.k,
            3 => self.real.w,
            4 => self.dual.i,
            5 => self.dual.j,
            6 => self.dual.k,
            7 => self.dual.w,
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
