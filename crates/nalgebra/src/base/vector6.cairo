//! `Vector6`: a statically sized 6-dimensional column vector (upstream `nalgebra::Vector6`),
//! stored as two `Vector3` blocks (DESIGN D4: the spatial-algebra layout rapier's multibody and
//! soft-body code uses, where a spatial vector is a pair of 3-vectors — angular and linear parts,
//! in whichever order the caller adopts).
//!
//! - `Vector6Trait` / `Vector6Impl`: constructors, block accessors, component-wise operations,
//!   reductions, the dot product, norms and interpolation, generic over a `simba::scalar::Real`
//!   scalar;
//! - operators `+`, `-`, unary `-`, `+=`, `-=` between vectors, `*=` and `/=` by a scalar, and
//!   conversions from / to `(Vector3, Vector3)` and `[T; 6]`: their impls live in this module,
//!   where the compiler finds them without any import.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel. The
//! 6-term sums (`dot`, `norm_squared`, `norm`) use the explicit `Real::Wide` accumulator, so they
//! cost ONE floor rounding and ONE overflow check, exactly like the 3-term `sum_prod3` kernels —
//! never two chained `sum_prod3`, which would round twice.

use core::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use simba::scalar::Real;
use super::vector3::{Vector3, Vector3Trait};

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;

/// A 6-dimensional column vector, as two 3-dimensional blocks.
///
/// Upstream `Vector6` names its six components `x, y, z, w, a, b`; here they live in the blocks:
/// `x, y, z` are `self.a.x, self.a.y, self.a.z` and `w, a, b` are `self.b.x, self.b.y, self.b.z`.
/// `new` still takes them in the upstream order, and `head` / `tail` return the blocks (upstream
/// `fixed_rows::<3>(0)` / `fixed_rows::<3>(3)`).
///
/// `Serde` writes the first block then the second, which for a column vector is exactly upstream's
/// storage order.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Vector6<T> {
    /// Rows 1 to 3 (upstream components `x, y, z`).
    pub a: Vector3<T>,
    /// Rows 4 to 6 (upstream components `w, a, b`).
    pub b: Vector3<T>,
}

/// Operations of `Vector6<T>` over a `Real` scalar. By value, unrolled, no loop.
pub trait Vector6Trait<T> {
    /// The vector `(x, y, z, w, a, b)`, in upstream component order. Upstream: `Vector6::new`.
    fn new(x: T, y: T, z: T, w: T, a: T, b: T) -> Vector6<T>;
    /// The zero vector. Upstream: `Vector6::zeros`.
    fn zeros() -> Vector6<T>;
    /// The vector whose rows 1 to 3 are `a` and whose rows 4 to 6 are `b`. Upstream: the two
    /// halves of `Vector6::from_iterator` / `fixed_rows_mut` assignments; rapier builds spatial
    /// vectors this way.
    fn from_blocks(a: Vector3<T>, b: Vector3<T>) -> Vector6<T>;
    /// Rows 1 to 3. Upstream: `fixed_rows::<3>(0)`.
    fn head(self: Vector6<T>) -> Vector3<T>;
    /// Rows 4 to 6. Upstream: `fixed_rows::<3>(3)`.
    fn tail(self: Vector6<T>) -> Vector3<T>;
    /// `self * k`, each component floored once. Panics on overflow. Upstream: `scale`
    /// (`self * k`).
    fn scale(self: Vector6<T>, k: T) -> Vector6<T>;
    /// `self / k`, each component being the exactly floored quotient. Panics on a zero `k` and on
    /// overflow. Upstream: `unscale` (`self / k`).
    ///
    /// One division per component on purpose, like `Vector3::unscale`: `scale(k.recip())` is
    /// cheaper but rounds `1 / k` first, which costs up to `|self|` ulp instead of 1.
    fn unscale(self: Vector6<T>, k: T) -> Vector6<T>;
    /// Component-wise product, each component floored once. Panics on overflow. Upstream:
    /// `component_mul`.
    fn component_mul(self: Vector6<T>, rhs: Vector6<T>) -> Vector6<T>;
    /// Component-wise absolute value. Exact; panics on overflow (`|MIN|`). Upstream: `abs`.
    fn abs(self: Vector6<T>) -> Vector6<T>;
    /// Component-wise minimum (infimum). Exact. Upstream: `inf`.
    fn inf(self: Vector6<T>, other: Vector6<T>) -> Vector6<T>;
    /// Component-wise maximum (supremum). Exact. Upstream: `sup`.
    fn sup(self: Vector6<T>, other: Vector6<T>) -> Vector6<T>;
    /// Sum of the six components. Exact; panics on overflow. Upstream: `sum`.
    fn sum(self: Vector6<T>) -> T;
    /// Dot product of the six components: the products are accumulated EXACTLY in the `Real::Wide`
    /// accumulator and rescaled ONCE (one floor, one overflow check), so the result is the exact
    /// floor of the mathematical dot product. Only the result must fit. Upstream: `dot`.
    ///
    /// Two chained `sum_prod3` (`head.dot(rhs.head) + tail.dot(rhs.tail)`) would round twice — up
    /// to 1 ulp off the exact floor AND 1.87x dearer (5 140 against 2 750 net), because a
    /// `sum_prod3` pays a full rescale where an extra `wide_add_prod` costs about 200 gas. Kept as
    /// `bench_vector6_dot__alt_two_sum_prod3`; one rounded product per term is 5.35x dearer
    /// (`__alt_unfused`).
    fn dot(self: Vector6<T>, rhs: Vector6<T>) -> T;
    /// Squared Euclidean norm: the six squares accumulated exactly, floored once. Panics on
    /// overflow — above a norm of about 46 340 (Q32.32) only `norm` works. Upstream:
    /// `norm_squared`.
    fn norm_squared(self: Vector6<T>) -> T;
    /// Euclidean norm: square root of the UNSCALED exact sum of squares (`Real::wide_sqrt`),
    /// floored once. No intermediate overflow: only the result must fit, so the norm of
    /// `(1e6, .., 1e6)` is fine. Upstream: `norm`.
    fn norm(self: Vector6<T>) -> T;
    /// `self / self.norm()`: the floored norm, then one exactly floored division per component
    /// (`unscale`). The error is about `1 + 1 / norm` ulp per component whatever the magnitude of
    /// `self`. Panics with a division by zero when the norm is zero, and on overflow when the norm
    /// does not fit. Upstream: `normalize`.
    fn normalize(self: Vector6<T>) -> Vector6<T>;
    /// `self + (rhs - self) * t` per component (`Real::lerp`: exact difference and product, one
    /// floor rounding). `t` is not clamped; `t = 0` gives `self` and `t = 1` gives `rhs` exactly.
    /// Panics on overflow of the result. Upstream: `lerp` (`self * (1 - t) + rhs * t`).
    fn lerp(self: Vector6<T>, rhs: Vector6<T>, t: T) -> Vector6<T>;
    /// `true` when every component is within `ulps` smallest units (raw units for fixed point) of
    /// the matching component of `other`; cannot overflow. Upstream:
    /// `approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float
    /// epsilon (DESIGN D3).
    fn abs_diff_eq(self: Vector6<T>, other: Vector6<T>, ulps: u64) -> bool;
}

pub impl Vector6Impl<
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
    +Drop<R::Wide>,
> of Vector6Trait<T> {
    #[inline(always)]
    fn new(x: T, y: T, z: T, w: T, a: T, b: T) -> Vector6<T> {
        Vector6 { a: Vector3 { x, y, z }, b: Vector3 { x: w, y: a, z: b } }
    }

    #[inline(always)]
    fn zeros() -> Vector6<T> {
        Vector6 {
            a: Vector3 { x: R::ZERO, y: R::ZERO, z: R::ZERO },
            b: Vector3 { x: R::ZERO, y: R::ZERO, z: R::ZERO },
        }
    }

    #[inline(always)]
    fn from_blocks(a: Vector3<T>, b: Vector3<T>) -> Vector6<T> {
        Vector6 { a, b }
    }

    #[inline(always)]
    fn head(self: Vector6<T>) -> Vector3<T> {
        self.a
    }

    #[inline(always)]
    fn tail(self: Vector6<T>) -> Vector3<T> {
        self.b
    }

    #[inline(always)]
    fn scale(self: Vector6<T>, k: T) -> Vector6<T> {
        Vector6 { a: Vector3Trait::scale(self.a, k), b: Vector3Trait::scale(self.b, k) }
    }

    #[inline(always)]
    fn unscale(self: Vector6<T>, k: T) -> Vector6<T> {
        Vector6 { a: Vector3Trait::unscale(self.a, k), b: Vector3Trait::unscale(self.b, k) }
    }

    #[inline(always)]
    fn component_mul(self: Vector6<T>, rhs: Vector6<T>) -> Vector6<T> {
        Vector6 {
            a: Vector3Trait::component_mul(self.a, rhs.a),
            b: Vector3Trait::component_mul(self.b, rhs.b),
        }
    }

    #[inline(always)]
    fn abs(self: Vector6<T>) -> Vector6<T> {
        Vector6 { a: Vector3Trait::abs(self.a), b: Vector3Trait::abs(self.b) }
    }

    #[inline(always)]
    fn inf(self: Vector6<T>, other: Vector6<T>) -> Vector6<T> {
        Vector6 { a: Vector3Trait::inf(self.a, other.a), b: Vector3Trait::inf(self.b, other.b) }
    }

    #[inline(always)]
    fn sup(self: Vector6<T>, other: Vector6<T>) -> Vector6<T> {
        Vector6 { a: Vector3Trait::sup(self.a, other.a), b: Vector3Trait::sup(self.b, other.b) }
    }

    #[inline(always)]
    fn sum(self: Vector6<T>) -> T {
        self.a.x + self.a.y + self.a.z + self.b.x + self.b.y + self.b.z
    }

    #[inline(always)]
    fn dot(self: Vector6<T>, rhs: Vector6<T>) -> T {
        let w = R::wide_add_prod(R::wide_zero(), self.a.x, rhs.a.x);
        let w = R::wide_add_prod(w, self.a.y, rhs.a.y);
        let w = R::wide_add_prod(w, self.a.z, rhs.a.z);
        let w = R::wide_add_prod(w, self.b.x, rhs.b.x);
        let w = R::wide_add_prod(w, self.b.y, rhs.b.y);
        R::wide_rescale(R::wide_add_prod(w, self.b.z, rhs.b.z))
    }

    #[inline(always)]
    fn norm_squared(self: Vector6<T>) -> T {
        let w = R::wide_add_prod(R::wide_zero(), self.a.x, self.a.x);
        let w = R::wide_add_prod(w, self.a.y, self.a.y);
        let w = R::wide_add_prod(w, self.a.z, self.a.z);
        let w = R::wide_add_prod(w, self.b.x, self.b.x);
        let w = R::wide_add_prod(w, self.b.y, self.b.y);
        R::wide_rescale(R::wide_add_prod(w, self.b.z, self.b.z))
    }

    #[inline(always)]
    fn norm(self: Vector6<T>) -> T {
        let w = R::wide_add_prod(R::wide_zero(), self.a.x, self.a.x);
        let w = R::wide_add_prod(w, self.a.y, self.a.y);
        let w = R::wide_add_prod(w, self.a.z, self.a.z);
        let w = R::wide_add_prod(w, self.b.x, self.b.x);
        let w = R::wide_add_prod(w, self.b.y, self.b.y);
        R::wide_sqrt(R::wide_add_prod(w, self.b.z, self.b.z))
    }

    #[inline(always)]
    fn normalize(self: Vector6<T>) -> Vector6<T> {
        Self::unscale(self, Self::norm(self))
    }

    #[inline(always)]
    fn lerp(self: Vector6<T>, rhs: Vector6<T>, t: T) -> Vector6<T> {
        Vector6 { a: Vector3Trait::lerp(self.a, rhs.a, t), b: Vector3Trait::lerp(self.b, rhs.b, t) }
    }

    #[inline(always)]
    fn abs_diff_eq(self: Vector6<T>, other: Vector6<T>, ulps: u64) -> bool {
        Vector3Trait::abs_diff_eq(self.a, other.a, ulps)
            && Vector3Trait::abs_diff_eq(self.b, other.b, ulps)
    }
}

/// `lhs + rhs`, component-wise. Exact; panics on overflow. Upstream: `Add`.
pub impl Vector6Add<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<Vector6<T>> {
    #[inline(always)]
    fn add(lhs: Vector6<T>, rhs: Vector6<T>) -> Vector6<T> {
        Vector6 { a: lhs.a + rhs.a, b: lhs.b + rhs.b }
    }
}

/// `lhs - rhs`, component-wise. Exact; panics on overflow. Upstream: `Sub`.
pub impl Vector6Sub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<Vector6<T>> {
    #[inline(always)]
    fn sub(lhs: Vector6<T>, rhs: Vector6<T>) -> Vector6<T> {
        Vector6 { a: lhs.a - rhs.a, b: lhs.b - rhs.b }
    }
}

/// `-a`, component-wise. Exact; panics on overflow (`-MIN`). Upstream: `Neg`.
pub impl Vector6Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Vector6<T>> {
    #[inline(always)]
    fn neg(a: Vector6<T>) -> Vector6<T> {
        Vector6 { a: -a.a, b: -a.b }
    }
}

/// `self += rhs`. Exact; panics on overflow. Upstream: `AddAssign`.
pub impl Vector6AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Vector6<T>, Vector6<T>> {
    #[inline(always)]
    fn add_assign(ref self: Vector6<T>, rhs: Vector6<T>) {
        self = Vector6 { a: self.a + rhs.a, b: self.b + rhs.b };
    }
}

/// `self -= rhs`. Exact; panics on overflow. Upstream: `SubAssign`.
pub impl Vector6SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Vector6<T>, Vector6<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Vector6<T>, rhs: Vector6<T>) {
        self = Vector6 { a: self.a - rhs.a, b: self.b - rhs.b };
    }
}

/// `self *= k` for a scalar `k`: `scale` in place (corelib's binary `*` is homogeneous, so `v * k`
/// is the named method `scale`). Upstream: `MulAssign<T>`.
pub impl Vector6MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<Vector6<T>, T> {
    #[inline(always)]
    fn mul_assign(ref self: Vector6<T>, rhs: T) {
        self =
            Vector6 {
                a: Vector3 { x: self.a.x * rhs, y: self.a.y * rhs, z: self.a.z * rhs },
                b: Vector3 { x: self.b.x * rhs, y: self.b.y * rhs, z: self.b.z * rhs },
            };
    }
}

/// `self /= k` for a scalar `k`: `unscale` in place. Upstream: `DivAssign<T>`.
pub impl Vector6DivAssign<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of DivAssign<Vector6<T>, T> {
    #[inline(always)]
    fn div_assign(ref self: Vector6<T>, rhs: T) {
        self =
            Vector6 {
                a: Vector3 {
                    x: R::div(self.a.x, rhs), y: R::div(self.a.y, rhs), z: R::div(self.a.z, rhs),
                },
                b: Vector3 {
                    x: R::div(self.b.x, rhs), y: R::div(self.b.y, rhs), z: R::div(self.b.z, rhs),
                },
            };
    }
}

/// `(head, tail).into()`: the two blocks as a `Vector6`.
pub impl Vector6FromBlocks<T> of Into<(Vector3<T>, Vector3<T>), Vector6<T>> {
    #[inline(always)]
    fn into(self: (Vector3<T>, Vector3<T>)) -> Vector6<T> {
        let (a, b) = self;
        Vector6 { a, b }
    }
}

/// The two blocks as a tuple `(head, tail)`.
pub impl Vector6IntoBlocks<T> of Into<Vector6<T>, (Vector3<T>, Vector3<T>)> {
    #[inline(always)]
    fn into(self: Vector6<T>) -> (Vector3<T>, Vector3<T>) {
        let Vector6 { a, b } = self;
        (a, b)
    }
}

/// `[x, y, z, w, a, b].into()`, in upstream component order. Upstream: `From<[T; 6]>`.
pub impl Vector6FromArray<T> of Into<[T; 6], Vector6<T>> {
    #[inline(always)]
    fn into(self: [T; 6]) -> Vector6<T> {
        let [x, y, z, w, a, b] = self;
        Vector6 { a: Vector3 { x, y, z }, b: Vector3 { x: w, y: a, z: b } }
    }
}

/// The components as a fixed-size array `[x, y, z, w, a, b]`. Upstream: `Into<[T; 6]>`.
pub impl Vector6IntoArray<T> of Into<Vector6<T>, [T; 6]> {
    #[inline(always)]
    fn into(self: Vector6<T>) -> [T; 6] {
        let Vector6 { a, b } = self;
        [a.x, a.y, a.z, b.x, b.y, b.z]
    }
}
