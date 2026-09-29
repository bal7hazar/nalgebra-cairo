//! Internal, no stability promise: the crate-private items of `linalg::symmetric_eigen3` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use core::internal::revoke_ap_tracking;
use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::internal::base::sym_matrix3::{SymMatrix3, SymMatrix3Trait};
use nalgebra_static3::base::matrix3::Matrix3Trait;
use nalgebra_static3::base::vector3::Vector3Trait;
use simba::scalar::Real;
use crate::linalg::symmetric_eigen3::SymmetricEigen3;

/// The running state of the Jacobi iteration: the partially diagonalised matrix `s` and the
/// accumulated rotation `v` (`s = vᵀ * original * v`). Private: `SymmetricEigen3` is the API.
#[derive(Copy, Drop)]
pub struct Jacobi3<T> {
    s: SymMatrix3<T>,
    v: Matrix3<T>,
}

/// The three plane rotations of one cyclic Jacobi sweep, plus the finishing step.
#[generate_trait]
pub impl Jacobi3Impl<
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
> of Jacobi3Trait<T> {
    /// The state `(s, I)` the sweeps start from.
    #[inline(always)]
    fn start(s: SymMatrix3<T>) -> Jacobi3<T> {
        Jacobi3 { s, v: Matrix3Trait::identity() }
    }

    /// `(c, s)` of the plane rotation that annihilates the off-diagonal entry `g = a_pq` of the
    /// 2x2 block `[[a_pp, g], [g, a_qq]]`.
    ///
    /// The textbook form is `t = sgn(θ) / (|θ| + sqrt(θ² + 1))` with `θ = (a_qq - a_pp) / (2
    /// g)`, then `c = 1 / sqrt(t² + 1)`, `s = t c`. `θ` itself is unrepresentable as soon as `g`
    /// is small, so the identical expression is used in the scale-free form obtained by multiplying
    /// numerator and denominator by `|g|`:
    ///
    /// ```text
    /// h = (a_qq - a_pp) / 2      t = sgn(h) * g / (|h| + sqrt(h² + g²))
    /// ```
    ///
    /// `sqrt(h² + g²)` is `Real::norm2`, whose sum of squares is accumulated unscaled: neither
    /// the square nor the ratio can overflow, and `|t| <= 1` always (`h = 0` gives `t = ±1`
    /// exactly, the 45° rotation). `c = 1 / sqrt(1 + t²)` is `recip(sqrt(mul_add(t, t, 1)))`: `1
    /// + t² <= 2`
    /// is exact to its last bit before the floored root, then one reciprocal rounded to nearest.
    /// Measured against `inv_norm2(1, t)` (simba bench helper: the floored norm and a 96-bit
    /// reciprocal, 3 020 gas cheaper per rotation): same eigen records, better SVD records (worst
    /// singular value 71 ulp instead of 83), so the dearer and more accurate form is kept. Five
    /// roundings in total: `h`, `t`, the root, `c`, `s`. The tangent is returned as well, because
    /// the diagonal update uses it directly. No upstream equivalent (`GivensRotation::new` solves a
    /// different problem).
    #[inline(always)]
    fn rotation(app: T, g: T, aqq: T) -> (T, T, T) {
        let h = R::diff_prod(aqq, R::HALF, app, R::HALF);
        let num = if h.is_sign_negative() {
            -g
        } else {
            g
        };
        let t = R::div(num, h.abs() + R::norm2(h, g));
        let c = R::recip(R::sqrt(R::mul_add(t, t, R::one())));
        (t, c, t * c)
    }

    /// One cyclic sweep: the rotations annihilating `m12`, then `m13`, then `m23`.
    fn sweep(self: Jacobi3<T>) -> Jacobi3<T> {
        Self::rotate23(Self::rotate13(Self::rotate12(self)))
    }

    /// One cyclic sweep on the matrix alone, dropping the rotation: what
    /// `SymmetricEigen3::eigenvalues` runs. The 6 fused products of the eigenvector update per
    /// rotation are 41 % of the cost of a sweep (measured: 125 380 gas with `v`, 73 460 without).
    #[inline(always)]
    fn sweep_s(s: SymMatrix3<T>) -> SymMatrix3<T> {
        let (s, _, _) = Self::rotate12_s(s);
        let (s, _, _) = Self::rotate13_s(s);
        let (s, _, _) = Self::rotate23_s(s);
        s
    }

    /// `(a, b, c)` sorted ascending: the sorting network of 3 elements, three conditional swaps.
    /// Branches are cheap; a `Vector3` of eigenvalues is never large enough to justify anything
    /// smarter.
    #[inline(always)]
    fn sorted_triple(a: T, b: T, c: T) -> Vector3<T> {
        let (mut l1, mut l2, mut l3) = (a, b, c);
        if l2 < l1 {
            let t = l1;
            l1 = l2;
            l2 = t;
        }
        if l3 < l1 {
            let t = l1;
            l1 = l3;
            l3 = t;
        }
        if l3 < l2 {
            let t = l2;
            l2 = l3;
            l3 = t;
        }
        Vector3 { x: l1, y: l2, z: l3 }
    }

    /// The rotation in the `(1, 2)` plane that annihilates `m12`.
    ///
    /// The diagonal update is the classical `a_pp -= t g`, `a_qq += t g` (one fused `mul_add`
    /// each, and an exact trace to within its rounding) rather than the quadratic form
    /// `c² a_pp - 2sc g + s² a_qq`; both were modelled bit-exactly, the `t` form is three times
    /// cheaper and measured 4x more accurate on the oracle suite (603 ulp against 2 465 ulp worst
    /// case).
    ///
    /// When `m12` is exactly zero the rotation is the identity and the update is skipped. That
    /// branch is a **correctness** guard, not a saving: `h` and `g` would both be zero on an
    /// isotropic 2x2 block and `t` would divide by zero. Sierra charges the maximum of the two
    /// branches, so taking it costs exactly as much as the rotation — measured, a diagonal input
    /// costs the same 551 k gas as a generic one (`bench_symmetric_eigen3_new__diagonal_input`).
    /// What it does buy is exactness: diagonal and isotropic matrices come out bit-exact.
    ///
    /// The matrix update is split from the eigenvector update so that `SymmetricEigen3::
    /// eigenvalues` can skip the latter. `#[inline(always)]` is against the rule for a kernel
    /// this size, but it is what makes the returned tuple disappear at the two call sites: it
    /// buys 23 520 gas on `new` (measured), which is why it stays.
    #[inline(always)]
    fn rotate12_s(s: SymMatrix3<T>) -> (SymMatrix3<T>, T, T) {
        if s.m12 == R::zero() {
            return (s, R::one(), R::zero());
        }
        let (t, c, sn) = Self::rotation(s.m11, s.m12, s.m22);
        (
            SymMatrix3 {
                m11: R::mul_add(-t, s.m12, s.m11),
                m12: R::zero(),
                m13: R::diff_prod(c, s.m13, sn, s.m23),
                m22: R::mul_add(t, s.m12, s.m22),
                m23: R::sum_prod2(sn, s.m13, c, s.m23),
                m33: s.m33,
            },
            c,
            sn,
        )
    }

    /// `rotate12_s` with the matching update of the accumulated rotation. The identity
    /// `(c, s) = (1, 0)` leaves `v` bit-identical, so the early exit of `rotate12_s` needs no
    /// counterpart here.
    fn rotate12(self: Jacobi3<T>) -> Jacobi3<T> {
        let v = self.v;
        let (s, c, sn) = Self::rotate12_s(self.s);
        Jacobi3 {
            s,
            v: Matrix3 {
                m11: R::diff_prod(c, v.m11, sn, v.m12),
                m12: R::sum_prod2(sn, v.m11, c, v.m12),
                m13: v.m13,
                m21: R::diff_prod(c, v.m21, sn, v.m22),
                m22: R::sum_prod2(sn, v.m21, c, v.m22),
                m23: v.m23,
                m31: R::diff_prod(c, v.m31, sn, v.m32),
                m32: R::sum_prod2(sn, v.m31, c, v.m32),
                m33: v.m33,
            },
        }
    }

    /// The rotation in the `(1, 3)` plane that annihilates `m13`, see `rotate12_s`.
    #[inline(always)]
    fn rotate13_s(s: SymMatrix3<T>) -> (SymMatrix3<T>, T, T) {
        if s.m13 == R::zero() {
            return (s, R::one(), R::zero());
        }
        let (t, c, sn) = Self::rotation(s.m11, s.m13, s.m33);
        (
            SymMatrix3 {
                m11: R::mul_add(-t, s.m13, s.m11),
                m12: R::diff_prod(c, s.m12, sn, s.m23),
                m13: R::zero(),
                m22: s.m22,
                m23: R::sum_prod2(sn, s.m12, c, s.m23),
                m33: R::mul_add(t, s.m13, s.m33),
            },
            c,
            sn,
        )
    }

    /// `rotate13_s` with the matching update of the accumulated rotation, see `rotate12`.
    fn rotate13(self: Jacobi3<T>) -> Jacobi3<T> {
        let v = self.v;
        let (s, c, sn) = Self::rotate13_s(self.s);
        Jacobi3 {
            s,
            v: Matrix3 {
                m11: R::diff_prod(c, v.m11, sn, v.m13),
                m12: v.m12,
                m13: R::sum_prod2(sn, v.m11, c, v.m13),
                m21: R::diff_prod(c, v.m21, sn, v.m23),
                m22: v.m22,
                m23: R::sum_prod2(sn, v.m21, c, v.m23),
                m31: R::diff_prod(c, v.m31, sn, v.m33),
                m32: v.m32,
                m33: R::sum_prod2(sn, v.m31, c, v.m33),
            },
        }
    }

    /// The rotation in the `(2, 3)` plane that annihilates `m23`, see `rotate12_s`.
    #[inline(always)]
    fn rotate23_s(s: SymMatrix3<T>) -> (SymMatrix3<T>, T, T) {
        if s.m23 == R::zero() {
            return (s, R::one(), R::zero());
        }
        let (t, c, sn) = Self::rotation(s.m22, s.m23, s.m33);
        (
            SymMatrix3 {
                m11: s.m11,
                m12: R::diff_prod(c, s.m12, sn, s.m13),
                m13: R::sum_prod2(sn, s.m12, c, s.m13),
                m22: R::mul_add(-t, s.m23, s.m22),
                m23: R::zero(),
                m33: R::mul_add(t, s.m23, s.m33),
            },
            c,
            sn,
        )
    }

    /// `rotate23_s` with the matching update of the accumulated rotation, see `rotate12`.
    fn rotate23(self: Jacobi3<T>) -> Jacobi3<T> {
        let v = self.v;
        let (s, c, sn) = Self::rotate23_s(self.s);
        Jacobi3 {
            s,
            v: Matrix3 {
                m11: v.m11,
                m12: R::diff_prod(c, v.m12, sn, v.m13),
                m13: R::sum_prod2(sn, v.m12, c, v.m13),
                m21: v.m21,
                m22: R::diff_prod(c, v.m22, sn, v.m23),
                m23: R::sum_prod2(sn, v.m22, c, v.m23),
                m31: v.m31,
                m32: R::diff_prod(c, v.m32, sn, v.m33),
                m33: R::sum_prod2(sn, v.m32, c, v.m33),
            },
        }
    }

    /// The diagonal of `s` sorted ascending, with the matching columns of `v`. The off-diagonal
    /// entries of `s` are dropped (measured below 1 ulp per unit of `max |m_ij|` after four
    /// sweeps). Three conditional swaps, the sorting network of 3 elements; branches are cheap.
    fn sorted(self: Jacobi3<T>) -> (Vector3<T>, Matrix3<T>) {
        let (mut l1, mut l2, mut l3) = (self.s.m11, self.s.m22, self.s.m33);
        let (mut c1, mut c2, mut c3) = (self.v.column1(), self.v.column2(), self.v.column3());
        if l2 < l1 {
            let (tl, tc) = (l1, c1);
            l1 = l2;
            c1 = c2;
            l2 = tl;
            c2 = tc;
        }
        if l3 < l1 {
            let (tl, tc) = (l1, c1);
            l1 = l3;
            c1 = c3;
            l3 = tl;
            c3 = tc;
        }
        if l3 < l2 {
            let (tl, tc) = (l2, c2);
            l2 = l3;
            c2 = c3;
            l3 = tl;
            c3 = tc;
        }
        (Vector3 { x: l1, y: l2, z: l3 }, Matrix3Trait::from_columns(c1, c2, c3))
    }

    /// The sorted state under the sign convention of `SymmetricEigen3`: columns 1 and 2
    /// renormalised and canonically signed, column 3 their cross product.
    ///
    /// The columns of `v` drift from unit norm by the rounding of the 12 rotations (measured: up
    /// to 15 ulp). Renormalising costs 2 `norm3` and 6 divisions — **22 460 gas, 4 % of `new`**
    /// —
    /// and buys 15 -> 11 ulp of orthonormality and a third off the reconstruction error
    /// (37 -> 26 ulp per unit of `max |m_ij|`); it is kept because the type promises unit columns
    /// and because the cross product of column 3 is only unit if its factors are. The variant
    /// that skips it is `bench_symmetric_eigen3_new__no_renormalisation`.
    fn finish(self: Jacobi3<T>) -> SymmetricEigen3<T> {
        let (eigenvalues, v) = self.sorted();
        let c1 = Self::canonical_sign(v.column1().normalize());
        let c2 = Self::canonical_sign(v.column2().normalize());
        SymmetricEigen3 {
            eigenvalues, eigenvectors: Matrix3Trait::from_columns(c1, c2, c1.cross(c2)),
        }
    }

    /// `v` with the sign that makes its component of largest absolute value positive (ties go to
    /// the earlier component); `v` unchanged when it is zero.
    #[inline(always)]
    fn canonical_sign(v: Vector3<T>) -> Vector3<T> {
        let (ax, ay, az) = (v.x.abs(), v.y.abs(), v.z.abs());
        let dominant = if ax >= ay && ax >= az {
            v.x
        } else if ay >= az {
            v.y
        } else {
            v.z
        };
        if dominant.is_sign_negative() {
            Vector3 { x: -v.x, y: -v.y, z: -v.z }
        } else {
            v
        }
    }
}

/// Crate-internal kernels of `SymmetricEigen3<T>` (WP 8.0: the public API is strictly
/// upstream's): the decomposition and the eigenvalues of a `SymMatrix3` (the 6 independent
/// components the SVD's Gram matrix is built as), the symmetric reconstruction.
#[generate_trait]
pub impl SymmetricEigen3InternalImpl<
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
> of SymmetricEigen3InternalTrait<T> {
    /// The kernel of `SymmetricEigen3Trait::new` on the 6 independent components of `s`, the
    /// form the SVD builds its Gram matrix in. Documented (cost, accuracy) on `new`.
    fn new_sym(s: SymMatrix3<T>) -> SymmetricEigen3<T> {
        // WP 8.5-P14b: the four sweeps push enough cells that a caller chaining a few
        // decompositions overflows the CASM offset of its live values (Sierra -> CASM "Offset
        // overflow"); an unknown ap change makes the callers spill them to locals first.
        revoke_ap_tracking();
        Jacobi3Impl::<T>::start(s).sweep().sweep().sweep().sweep().finish()
    }
    /// The kernel of `try_new` (WP 8.5-P14b): the same four sweeps, then upstream's convergence
    /// test on the final state — every off-diagonal entry within `eps * (|s_ii| + |s_jj|)` (one
    /// fused product pair, floored) — before `finish`. Bit-identical to `new_sym` when `Some`.
    fn try_new_sym(s: SymMatrix3<T>, eps: T) -> Option<SymmetricEigen3<T>> {
        revoke_ap_tracking();
        let j = Jacobi3Impl::<T>::start(s).sweep().sweep().sweep().sweep();
        let s = j.s;
        if s.m12.abs() <= R::sum_prod2(eps, s.m11.abs(), eps, s.m22.abs())
            && s.m13.abs() <= R::sum_prod2(eps, s.m11.abs(), eps, s.m33.abs())
            && s.m23.abs() <= R::sum_prod2(eps, s.m22.abs(), eps, s.m33.abs()) {
            Some(j.finish())
        } else {
            None
        }
    }
    /// The eigenvalues of `s` alone, ascending: the same four sweeps as `new`, with the
    /// accumulation of the rotation dropped. Bit-identical to `new(s).eigenvalues` (the rotation
    /// never feeds back into the matrix), and measured **45 % cheaper** (303 470 against 550 770
    /// gas): 6 fused products per
    /// rotation and the two final `norm3` / six divisions disappear.
    /// The kernel of `Matrix3SymmetricEigenTrait::symmetric_eigenvalues`.
    fn eigenvalues(s: SymMatrix3<T>) -> Vector3<T> {
        revoke_ap_tracking();
        let s = Jacobi3Impl::<T>::sweep_s(s);
        let s = Jacobi3Impl::<T>::sweep_s(s);
        let s = Jacobi3Impl::<T>::sweep_s(s);
        let s = Jacobi3Impl::<T>::sweep_s(s);
        Jacobi3Impl::<T>::sorted_triple(s.m11, s.m22, s.m33)
    }
    /// `V * diag(eigenvalues) * Vᵀ`, the symmetric matrix the decomposition came from, up to the
    /// rounding of the decomposition (measured: **31 ulp per unit of `max |m_ij|`**). Goes through
    /// `SymMatrix3::quadform`, so only the 6 independent components are computed. Panics on
    /// overflow. The kernel of `recompose`, which mirrors it.
    #[inline(always)]
    fn recompose_sym(self: SymmetricEigen3<T>) -> SymMatrix3<T> {
        SymMatrix3Trait::quadform(self.eigenvectors, self.eigenvalues)
    }
}
