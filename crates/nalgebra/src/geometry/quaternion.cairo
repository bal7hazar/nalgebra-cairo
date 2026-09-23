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
//! | glam, for glam.cairo's conversions | `Quat::from_xyzw(x, y, z, w)` = `(i, j, k, w)` |
//!
//! The unit quaternion of a 3D rotation is `UnitQuaternion` (`geometry::unit_quaternion`); this
//! type is the general algebra it is built on.
//!
//! **Not ported:** upstream's transcendental quaternion functions (`exp`, `ln`, `powf`, `sqrt`,
//! `polar_decomposition`, the hyperbolic and trigonometric families). `exp` costs one scalar `exp`,
//! one `sin_cos`, one norm and four divisions (about 45 000 gas) and `ln` as much, nothing in the
//! physics stack calls them (research/01 §3.3), and the rotation-specific cases are covered
//! without them: `UnitQuaternion::from_scaled_axis` IS `exp` of a pure quaternion, `scaled_axis` is
//! its `ln`, and `UnitQuaternion::powf` goes through the axis and the angle. Add them here if a
//! client ever needs the general algebra.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

use simba::scalar::Real;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

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
        Quaternion { i: R::ZERO, j: R::ZERO, k: R::ZERO, w: R::ONE }
    }

    /// The quaternion `0`. Upstream: `Quaternion::zero` (`Zero::zero`).
    #[inline(always)]
    fn zero() -> Quaternion<T> {
        Quaternion { i: R::ZERO, j: R::ZERO, k: R::ZERO, w: R::ZERO }
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
        Quaternion { i: vector.x, j: vector.y, k: vector.z, w: R::ZERO }
    }

    /// The real quaternion `(w, 0, 0, 0)`. Upstream: `Quaternion::from_real`.
    #[inline(always)]
    fn from_real(w: T) -> Quaternion<T> {
        Quaternion { i: R::ZERO, j: R::ZERO, k: R::ZERO, w }
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
        if n2 == R::ZERO {
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
    /// the matching component of `other`; cannot overflow. Note that `q` and `-q` are the same
    /// rotation but are NOT `abs_diff_eq` (see `UnitQuaternionTrait::angle_to` for a rotation
    /// distance). Upstream: `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp
    /// instead of a float epsilon (DESIGN D3).
    #[inline(always)]
    fn abs_diff_eq(self: Quaternion<T>, other: Quaternion<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.i, other.i, ulps)
            && R::abs_diff_eq(self.j, other.j, ulps)
            && R::abs_diff_eq(self.k, other.k, ulps)
            && R::abs_diff_eq(self.w, other.w, ulps)
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

/// `coords.into()`: the quaternion of coordinates `(i, j, k, w)` (the STORAGE order). Upstream:
/// `From<Vector4> for Quaternion`.
pub impl QuaternionFromVector<T> of Into<Vector4<T>, Quaternion<T>> {
    #[inline(always)]
    fn into(self: Vector4<T>) -> Quaternion<T> {
        let Vector4 { x, y, z, w } = self;
        Quaternion { i: x, j: y, k: z, w }
    }
}

/// The coordinates `(i, j, k, w)` (the STORAGE order). Upstream: the `coords` field.
pub impl QuaternionIntoVector<T> of Into<Quaternion<T>, Vector4<T>> {
    #[inline(always)]
    fn into(self: Quaternion<T>) -> Vector4<T> {
        let Quaternion { i, j, k, w } = self;
        Vector4 { x: i, y: j, z: k, w }
    }
}
