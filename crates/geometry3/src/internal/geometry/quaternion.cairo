//! In `nalgebra_geometry3`: the impl `QuaternionInternalImpl`. This module is split over packages;
//! the other parts are in `nalgebra_core`.
//!
//! Internal, no stability promise: the crate-private items of `geometry::quaternion` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::Real;
use crate::geometry::quaternion::Quaternion;

/// Crate-internal kernels of `Quaternion<T>` (WP 8.0: the public API is strictly upstream's): the
/// fused `a.conjugate() * b` of `UnitQuaternion::conj_mul` / `Isometry3::inv_mul` (WP 4.5).
#[generate_trait]
pub impl QuaternionInternalImpl<
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
    #[inline(always)]
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
