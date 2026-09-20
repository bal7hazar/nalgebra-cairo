//! `SymmetricEigen3`: eigenvalues and eigenvectors of a symmetric 3x3 matrix, by a **fixed-sweep
//! cyclic Jacobi** rotation (DESIGN D6).
//!
//! Why not upstream's algorithm: `nalgebra::linalg::SymmetricEigen` tridiagonalises and then runs
//! an implicitly shifted symmetric QR until `|off| <= eps * (|d_m| + |d_n|)`, bounded only by a
//! `max_niter` argument. An unbounded loop has no place in a proven program: its gas depends on
//! the data, and the tolerance has no meaning in fixed point.
//!
//! Why not the closed form: `glamx::SymmetricEigen3` (parry / rapier's 3x3) uses Eberly's
//! trigonometric method, which needs `acos` and `cos` — `simba` has no `Transcendental<Fixed>`
//! impl yet, and `docs/research/01-nalgebra-analysis.md` records that it misbehaves on isotropic
//! inertia tensors, exactly the input rapier feeds it. A Cardano form needs the same `acos`. Even
//! ignoring that, the coefficients of the characteristic polynomial are the bottleneck: forming
//! `tr`, the sum of the principal 2x2 minors and `det` with the fused kernels and then solving the
//! cubic *exactly in f64* still costs up to 90 ulp on `small` matrices (cancellation in `det`),
//! against 9 ulp for the Jacobi sweeps below. The measurement is in the report of this work
//! package; the closed form is not ported.
//!
//! Cyclic Jacobi is a sequence of plane rotations, each one annihilating one off-diagonal entry
//! and each one an exact similarity transform in exact arithmetic. It converges quadratically, it
//! needs only `sqrt` and division, it is unconditionally stable, and **four sweeps of the three
//! rotations drive every off-diagonal entry of a 3x3 matrix to exactly zero** in Q32.32 —
//! measured over the oracle suite and 2 800 random / degenerate matrices (see `new`). The sweep
//! count is a constant, so the gas of the decomposition is a constant.

use simba::scalar::Real;
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::sym_matrix3::{SymMatrix3, SymMatrix3Trait};
use crate::base::vector3::{Vector3, Vector3Trait};

/// The eigendecomposition `S = V * diag(eigenvalues) * Vᵀ` of a symmetric 3x3 matrix.
///
/// `eigenvalues` are sorted **ascending** (like `tools/oracle`, unlike upstream, which leaves the
/// order to the QR sweep); the columns of `eigenvectors` are the matching unit eigenvectors, in
/// the same order. `eigenvectors` is a rotation: `det(eigenvectors) = +1` by construction.
///
/// Sign convention (upstream has none; eigenvectors are only defined up to a sign):
/// **columns 1 and 2 are oriented so that their component of largest absolute value is positive**
/// (ties go to the earlier component, `x` before `y` before `z`), and **column 3 is their cross
/// product** — which is the eigenvector of the largest eigenvalue up to rounding, and makes the
/// determinant `+1` without a sign test. The decomposition of a given matrix is therefore a
/// deterministic function of its raw components, as required by AGENTS.md.
///
/// Upstream: `SymmetricEigen { eigenvalues: OVector, eigenvectors: OMatrix }`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct SymmetricEigen3<T> {
    /// The three eigenvalues, ascending.
    pub eigenvalues: Vector3<T>,
    /// The matching unit eigenvectors, as columns (`det = +1`).
    pub eigenvectors: Matrix3<T>,
}

/// The running state of the Jacobi iteration: the partially diagonalised matrix `s` and the
/// accumulated rotation `v` (`s = vᵀ * original * v`). Private: `SymmetricEigen3` is the API.
#[derive(Copy, Drop)]
struct Jacobi3<T> {
    s: SymMatrix3<T>,
    v: Matrix3<T>,
}

/// The three plane rotations of one cyclic Jacobi sweep, plus the finishing step.
#[generate_trait]
impl Jacobi3Impl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Div<T>,
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
    /// exactly, the 45° rotation). `1 + t² <= 2`, so `inv_sqrt` is exact to its last bit. Four
    /// roundings in total: `h`, `t`, `c`, `s`. The tangent is returned as well, because the
    /// diagonal update uses it directly. No upstream equivalent (`GivensRotation::new` solves a
    /// different problem).
    #[inline(always)]
    fn rotation(app: T, g: T, aqq: T) -> (T, T, T) {
        let h = R::diff_prod(aqq, R::HALF, app, R::HALF);
        let num = if h.is_negative() {
            -g
        } else {
            g
        };
        let t = num / (h.abs() + R::norm2(h, g));
        let c = R::mul_add(t, t, R::ONE).inv_sqrt();
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
        if s.m12 == R::ZERO {
            return (s, R::ONE, R::ZERO);
        }
        let (t, c, sn) = Self::rotation(s.m11, s.m12, s.m22);
        (
            SymMatrix3 {
                m11: R::mul_add(-t, s.m12, s.m11),
                m12: R::ZERO,
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
        if s.m13 == R::ZERO {
            return (s, R::ONE, R::ZERO);
        }
        let (t, c, sn) = Self::rotation(s.m11, s.m13, s.m33);
        (
            SymMatrix3 {
                m11: R::mul_add(-t, s.m13, s.m11),
                m12: R::diff_prod(c, s.m12, sn, s.m23),
                m13: R::ZERO,
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
        if s.m23 == R::ZERO {
            return (s, R::ONE, R::ZERO);
        }
        let (t, c, sn) = Self::rotation(s.m22, s.m23, s.m33);
        (
            SymMatrix3 {
                m11: s.m11,
                m12: R::diff_prod(c, s.m12, sn, s.m13),
                m13: R::sum_prod2(sn, s.m12, c, s.m13),
                m22: R::mul_add(-t, s.m23, s.m22),
                m23: R::ZERO,
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
        if dominant.is_negative() {
            Vector3 { x: -v.x, y: -v.y, z: -v.z }
        } else {
            v
        }
    }
}

/// Methods of `SymmetricEigen3<T>` for any `Real` scalar.
#[generate_trait]
pub impl SymmetricEigen3Impl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Div<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of SymmetricEigen3Trait<T> {
    /// The eigendecomposition of `s` by **four** cyclic Jacobi sweeps, unrolled.
    ///
    /// Cost: **constant**, 550 770 gas — 12 plane rotations, whatever the input. The early exit
    /// of `rotate12_s` on an exactly zero off-diagonal entry is a correctness guard, not a saving
    /// (Sierra charges the maximum of the two branches): a diagonal input costs the same as a
    /// generic one. There is no convergence test, no tolerance and no iteration count; see
    /// `docs/DESIGN.md` D6.
    ///
    /// **Convergence, measured** bit-exactly over the 60 oracle vectors of the
    /// `symmetric_eigen_svd` suite plus 2 800 random (`small` / `unit` / `medium` / `large`), SPD,
    /// clustered, isotropic, diagonal, rank-1 and 1-ulp-perturbed matrices. The first two columns
    /// are in ulp per unit of `max |m_ij|`, the third in per cent of the oracle tolerance:
    ///
    /// | sweeps | `max \|offdiag\|` left | `recompose` | eigenvalues | gas |
    /// |---|---|---|---|---|
    /// | 3 | 13 163 | 8 582 | 12.1 % | 425 190 |
    /// | **4** | **1** | **31** | **12.1 %** | **550 770** |
    /// | 5 | 0 | 31 | 12.1 % | 676 350 |
    /// | 6 | 0 | 31 | 12.1 % | 801 930 |
    ///
    /// Four sweeps is the smallest count that reaches the fixed point: a fifth sweep finds every
    /// off-diagonal entry already zero and returns the state unchanged, so it is pure cost. The
    /// eigenvalues stay 8x inside the oracle tolerance, and the residual bound to quote is
    /// **`|S - V diag(λ) Vᵀ| <= 31 ulp * max(1, max |m_ij|)`** with the columns unit and
    /// mutually orthogonal within **16 ulp**. The 3, 5 and 6 sweep variants are kept as benchmarks
    /// here.
    ///
    /// Rounding: 4 roundings per rotation for `(c, s)` and one per updated component. Panics on
    /// overflow; a rotation never grows a component by more than `sqrt(2)`, so an input whose
    /// entries fit with one bit to spare cannot overflow.
    /// Upstream: `SymmetricEigen::new` (iterative QR, different algorithm and different order).
    fn new(s: SymMatrix3<T>) -> SymmetricEigen3<T> {
        Jacobi3Impl::<T>::start(s).sweep().sweep().sweep().sweep().finish()
    }

    /// The eigendecomposition of `m`, **assumed symmetric**: only the upper triangle is read, the
    /// lower one is ignored. Upstream: `Matrix3::symmetric_eigen` (which likewise reads a single
    /// triangle).
    #[inline(always)]
    fn from_matrix(m: Matrix3<T>) -> SymmetricEigen3<T> {
        Self::new(SymMatrix3Trait::from_matrix_unchecked(m))
    }

    /// The eigenvalues of `s` alone, ascending: the same four sweeps as `new`, with the
    /// accumulation of the rotation dropped. Bit-identical to `new(s).eigenvalues` (the rotation
    /// never feeds back into the matrix), and measured **45 % cheaper** (303 470 against 550 770
    /// gas): 6 fused products per
    /// rotation and the two final `norm3` / six divisions disappear.
    /// Upstream: `Matrix3::symmetric_eigenvalues`.
    fn eigenvalues(s: SymMatrix3<T>) -> Vector3<T> {
        let s = Jacobi3Impl::<T>::sweep_s(s);
        let s = Jacobi3Impl::<T>::sweep_s(s);
        let s = Jacobi3Impl::<T>::sweep_s(s);
        let s = Jacobi3Impl::<T>::sweep_s(s);
        Jacobi3Impl::<T>::sorted_triple(s.m11, s.m22, s.m33)
    }

    /// `V * diag(eigenvalues) * Vᵀ`, the symmetric matrix the decomposition came from, up to the
    /// rounding of the decomposition (measured: **31 ulp per unit of `max |m_ij|`**). Goes through
    /// `SymMatrix3::quadform`, so only the 6 independent components are computed. Panics on
    /// overflow. Upstream: `SymmetricEigen::recompose` (which returns the full matrix, see
    /// `recompose_matrix`).
    #[inline(always)]
    fn recompose(self: SymmetricEigen3<T>) -> SymMatrix3<T> {
        SymMatrix3Trait::quadform(self.eigenvectors, self.eigenvalues)
    }

    /// `recompose()` as a full `Matrix3`, bit-identical to it by symmetry.
    /// Upstream: `SymmetricEigen::recompose`.
    #[inline(always)]
    fn recompose_matrix(self: SymmetricEigen3<T>) -> Matrix3<T> {
        Self::recompose(self).to_matrix()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra_testing::black_box;
    use simba::fixed::Fixed;
    use simba::scalar::Real;
    use crate::base::matrix3::{Matrix3, Matrix3Trait};
    use crate::base::sym_matrix3::{SymMatrix3, SymMatrix3Trait};
    use crate::base::vector3::{Vector3, Vector3Trait};
    use crate::linalg::eigen_test_utils::{
        amax_s3, fx, int, m3, max_ulp_diff_s3, max_ulp_diff_v3, s3, s3i, ulp_diff, v3, v3i,
    };
    use crate::linalg::oracle_symmetric_eigen;
    use super::{Jacobi3, Jacobi3Impl, Jacobi3Trait, SymmetricEigen3, SymmetricEigen3Trait};

    // --- the losing candidates of the sweep-count study (kept as evidence) ----------------------

    /// `new` with three sweeps: leaves off-diagonal entries behind
    /// (see `test_three_sweeps_do_not_converge`).
    fn eigen_sweeps3(s: SymMatrix3<Fixed>) -> SymmetricEigen3<Fixed> {
        Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().finish()
    }

    /// `new` with five sweeps: the fifth sweep finds every off-diagonal entry already zero and
    /// returns the state unchanged (see `test_a_fifth_and_sixth_sweep_change_nothing`).
    fn eigen_sweeps5(s: SymMatrix3<Fixed>) -> SymmetricEigen3<Fixed> {
        Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().sweep().sweep().finish()
    }

    /// `new` with six sweeps.
    fn eigen_sweeps6(s: SymMatrix3<Fixed>) -> SymmetricEigen3<Fixed> {
        Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().sweep().sweep().sweep().finish()
    }

    /// `finish` without the final renormalisation of the columns: 2 `norm3` and 6 divisions
    /// cheaper, and measurably less orthonormal.
    fn finish_without_renormalisation(j: Jacobi3<Fixed>) -> SymmetricEigen3<Fixed> {
        let (eigenvalues, v) = j.sorted();
        let c1 = Jacobi3Impl::<Fixed>::canonical_sign(v.column1());
        let c2 = Jacobi3Impl::<Fixed>::canonical_sign(v.column2());
        SymmetricEigen3 {
            eigenvalues, eigenvectors: Matrix3Trait::from_columns(c1, c2, c1.cross(c2)),
        }
    }

    // --- error measures ------------------------------------------------------------------------

    /// `|<c_i, c_j> - delta_ij|` over the three columns, in raw units.
    fn orthonormality_error(v: Matrix3<Fixed>) -> u128 {
        let (c1, c2, c3) = (v.column1(), v.column2(), v.column3());
        let mut e = ulp_diff(c1.norm(), Real::ONE);
        e = core::cmp::max(e, ulp_diff(c2.norm(), Real::ONE));
        e = core::cmp::max(e, ulp_diff(c3.norm(), Real::ONE));
        e = core::cmp::max(e, ulp_diff(c1.dot(c2), Real::ZERO));
        e = core::cmp::max(e, ulp_diff(c1.dot(c3), Real::ZERO));
        core::cmp::max(e, ulp_diff(c2.dot(c3), Real::ZERO))
    }

    /// `|S * c_i - lambda_i * c_i|` over the three columns, in raw units.
    fn residual_error(s: SymMatrix3<Fixed>, e: SymmetricEigen3<Fixed>) -> u128 {
        let (c1, c2, c3) = (
            e.eigenvectors.column1(), e.eigenvectors.column2(), e.eigenvectors.column3(),
        );
        let zero = Vector3Trait::zeros();
        let mut r = max_ulp_diff_v3(s.mul_vec(c1) - c1.scale(e.eigenvalues.x), zero);
        r = core::cmp::max(r, max_ulp_diff_v3(s.mul_vec(c2) - c2.scale(e.eigenvalues.y), zero));
        core::cmp::max(r, max_ulp_diff_v3(s.mul_vec(c3) - c3.scale(e.eigenvalues.z), zero))
    }

    /// Largest `|off-diagonal|` of the partially diagonalised matrix, in raw units.
    fn off_diagonal_error(j: Jacobi3<Fixed>) -> u128 {
        let mut e = ulp_diff(j.s.m12, Real::ZERO);
        e = core::cmp::max(e, ulp_diff(j.s.m13, Real::ZERO));
        core::cmp::max(e, ulp_diff(j.s.m23, Real::ZERO))
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_diagonal_is_exact() {
        let e = SymmetricEigen3Trait::new(s3i(-2, 0, 0, 1, 0, 7));
        assert!(e.eigenvalues == v3i(-2, 1, 7));
        assert!(e.eigenvectors == Matrix3Trait::identity());
        assert!(e.recompose() == s3i(-2, 0, 0, 1, 0, 7));
    }

    #[test]
    fn test_new_diagonal_permutations_are_exact_and_right_handed() {
        // Every permutation of (-2, 1, 7) on the diagonal: the columns are signed axes, the
        // eigenvalues ascend and the determinant is exactly +1.
        let mut cases = [(-2, 1, 7), (-2, 7, 1), (1, -2, 7), (1, 7, -2), (7, -2, 1), (7, 1, -2)]
            .span();
        while let Some(case) = cases.pop_front() {
            let (a, b, c) = *case;
            let s = s3i(a, 0, 0, b, 0, c);
            let e = SymmetricEigen3Trait::new(s);
            assert!(e.eigenvalues == v3i(-2, 1, 7));
            assert!(e.eigenvectors.determinant() == Real::ONE);
            assert!(e.recompose() == s);
            assert!(residual_error(s, e) == 0);
        }
    }

    #[test]
    fn test_new_isotropic_is_exact() {
        let e = SymmetricEigen3Trait::new(s3i(3, 0, 0, 3, 0, 3));
        assert!(e.eigenvalues == v3i(3, 3, 3));
        assert!(e.eigenvectors == Matrix3Trait::identity());
        assert!(e.recompose() == s3i(3, 0, 0, 3, 0, 3));
        // Zero is isotropic too, and must not divide by zero.
        let e = SymmetricEigen3Trait::new(s3i(0, 0, 0, 0, 0, 0));
        assert!(e.eigenvalues == v3i(0, 0, 0));
        assert!(e.eigenvectors == Matrix3Trait::identity());
    }

    #[test]
    fn test_new_repeated_eigenvalues_is_exact() {
        // diag(5, 3, 3): the (2, 3) eigenspace is a plane, any orthonormal basis of it is valid.
        let s = s3i(5, 0, 0, 3, 0, 3);
        let e = SymmetricEigen3Trait::new(s);
        assert!(e.eigenvalues == v3i(3, 3, 5));
        assert!(e.eigenvectors.determinant() == Real::ONE);
        assert!(residual_error(s, e) == 0);
        assert!(e.recompose() == s);
    }

    #[test]
    fn test_new_rank_one_matrix() {
        // The all-ones matrix: eigenvalues 0, 0, 3 with (1, 1, 1)/sqrt(3) for 3.
        let s = s3i(1, 1, 1, 1, 1, 1);
        let e = SymmetricEigen3Trait::new(s);
        assert!(max_ulp_diff_v3(e.eigenvalues, v3i(0, 0, 3)) <= 2);
        let third = Real::<Fixed>::from_int(3).inv_sqrt();
        let c3 = e.eigenvectors.column3();
        assert!(max_ulp_diff_v3(c3.abs(), Vector3 { x: third, y: third, z: third }) <= 4);
        assert!(orthonormality_error(e.eigenvectors) <= 16);
        assert!(residual_error(s, e) <= 8);
    }

    #[test]
    fn test_new_block_diagonal_case_is_exact() {
        // [[5, 2, 0], [2, 2, 0], [0, 0, 4]]: the 2x2 block has eigenvalues 1 and 6.
        let s = s3i(5, 2, 0, 2, 0, 4);
        let e = SymmetricEigen3Trait::new(s);
        assert!(e.eigenvalues == v3i(1, 4, 6));
        // The eigenvectors of the block are irrational, so `det` is only +1 up to their rounding.
        assert!(ulp_diff(e.eigenvectors.determinant(), Real::ONE) <= 4);
        assert!(residual_error(s, e) <= 8);
    }

    #[test]
    fn test_canonical_sign_picks_the_dominant_component() {
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(-3, 1, 2)) == v3i(3, -1, -2));
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(1, -3, 2)) == v3i(-1, 3, -2));
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(1, 2, -3)) == v3i(-1, -2, 3));
        // Ties go to the earlier component.
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(-1, 1, 1)) == v3i(1, -1, -1));
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(0, -1, 1)) == v3i(0, 1, -1));
        assert!(Jacobi3Impl::<Fixed>::canonical_sign(v3i(0, 0, 0)) == v3i(0, 0, 0));
    }

    #[test]
    fn test_from_matrix_reads_the_upper_triangle() {
        let m = Matrix3Trait::new(
            int(5), int(2), int(0), int(-9), int(2), int(0), int(-9), int(-9), int(4),
        );
        assert!(SymmetricEigen3Trait::from_matrix(m).eigenvalues == v3i(1, 4, 6));
    }

    #[test]
    fn test_eigenvalues_matches_new() {
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3(a);
            assert!(
                SymmetricEigen3Trait::eigenvalues(s) == SymmetricEigen3Trait::new(s).eigenvalues,
            );
        }
    }

    // --- sweep count ---------------------------------------------------------------------------

    #[test]
    fn test_four_sweeps_reach_the_fixed_point() {
        // Every off-diagonal entry is exactly zero after four sweeps, on every oracle vector.
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let j = Jacobi3Impl::<Fixed>::start(s3(a)).sweep().sweep().sweep().sweep();
            assert!(off_diagonal_error(j) == 0, "four sweeps left a non-zero off-diagonal entry");
        }
    }

    #[test]
    fn test_three_sweeps_do_not_converge() {
        // Evidence for the choice of 4: three sweeps leave a visible off-diagonal residue.
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let j = Jacobi3Impl::<Fixed>::start(s3(a)).sweep().sweep().sweep();
            worst = core::cmp::max(worst, off_diagonal_error(j) / amax_s3(s3(a)));
        }
        assert!(worst > 0, "three sweeps already converge: the fourth could be dropped");
    }

    #[test]
    fn test_a_fifth_and_sixth_sweep_change_nothing() {
        // Evidence that 4 is not merely sufficient but the fixed point: more sweeps are identical.
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3(a);
            let four = SymmetricEigen3Trait::new(s);
            assert!(eigen_sweeps5(s) == four, "a fifth sweep changed the result");
            assert!(eigen_sweeps6(s) == four, "a sixth sweep changed the result");
        }
    }

    #[test]
    fn test_three_sweeps_are_less_accurate() {
        // The worst reconstruction error over the oracle suite, 3 sweeps against 4.
        let mut worst3: u128 = 0;
        let mut worst4: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3(a);
            let scale = amax_s3(s);
            let e3 = max_ulp_diff_s3(eigen_sweeps3(s).recompose(), s) / scale;
            let e4 = max_ulp_diff_s3(SymmetricEigen3Trait::new(s).recompose(), s) / scale;
            worst3 = core::cmp::max(worst3, e3);
            worst4 = core::cmp::max(worst4, e4);
        }
        assert!(worst4 < worst3, "the fourth sweep did not improve the reconstruction");
        assert!(worst4 <= 26 && worst3 > 26, "worst case regressed");
    }

    #[test]
    fn test_renormalisation_improves_orthonormality() {
        let mut with: u128 = 0;
        let mut without: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3(a);
            let j = Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().sweep();
            with = core::cmp::max(with, orthonormality_error(j.finish().eigenvectors));
            without =
                core::cmp::max(
                    without, orthonormality_error(finish_without_renormalisation(j).eigenvectors),
                );
        }
        assert!(with <= without, "renormalisation made the columns less orthonormal");
        assert!(with <= 16, "worst case regressed");
    }

    // --- oracle --------------------------------------------------------------------------------

    #[test]
    fn test_new_eigenvalues_oracle() {
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = SymmetricEigen3Trait::new(s3(a)).eigenvalues;
            let e = max_ulp_diff_v3(got, v3(expected));
            assert!(e <= tol.into(), "eigenvalues off by more than the oracle tolerance");
            worst = core::cmp::max(worst, e);
        }
        // Measured: 603 ulp worst case, against an oracle tolerance of up to 12 652.
        assert!(worst <= 603, "worst case regressed");
    }

    #[test]
    fn test_new_eigenvalues_oracle_spd() {
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_spd_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = SymmetricEigen3Trait::new(s3(a)).eigenvalues;
            let e = max_ulp_diff_v3(got, v3(expected));
            assert!(e <= tol.into(), "eigenvalues off by more than the oracle tolerance");
            worst = core::cmp::max(worst, e);
        }
        assert!(worst <= 264, "worst case regressed");
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle() {
        let mut worst_rec: u128 = 0;
        let mut worst_orth: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3(a);
            let e = SymmetricEigen3Trait::new(s);
            assert!(ulp_diff(e.eigenvectors.determinant(), Real::ONE) <= 4);
            let orth = orthonormality_error(e.eigenvectors);
            assert!(orth <= 32, "columns are not orthonormal");
            let rec = max_ulp_diff_s3(e.recompose(), s) / amax_s3(s);
            assert!(rec <= 32, "reconstruction is off");
            assert!(e.recompose_matrix() == e.recompose().to_matrix());
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        // Measured worst cases over the 30 vectors.
        assert!(worst_rec <= 26 && worst_orth <= 16, "worst case regressed");
    }

    #[test]
    fn test_new_recompose_oracle_spd() {
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_spd_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s3(a);
            let e = SymmetricEigen3Trait::new(s);
            assert!(max_ulp_diff_s3(e.recompose(), s) / amax_s3(s) <= 32);
            assert!(orthonormality_error(e.eigenvectors) <= 32);
            assert!(residual_error(s, e) <= 32 * amax_s3(s));
        }
    }

    #[test]
    fn test_new_eigenvectors_match_the_matrix_form() {
        let mut cases = oracle_symmetric_eigen::symmetric_eigen3_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            assert!(SymmetricEigen3Trait::from_matrix(m3(a)) == SymmetricEigen3Trait::new(s3(a)));
        }
    }

    // --- overflow ------------------------------------------------------------------------------

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_new_overflow_panics() {
        let s = black_box(
            SymMatrix3 {
                m11: Real::<Fixed>::MAX,
                m12: Real::MAX,
                m13: Real::MAX,
                m22: Real::MAX,
                m23: Real::MAX,
                m33: Real::MAX,
            },
        );
        SymmetricEigen3Trait::new(s);
    }

    // --- gas -----------------------------------------------------------------------------------

    /// A generic symmetric matrix: no off-diagonal entry is zero, so no rotation is skipped.
    fn bench_input() -> SymMatrix3<Fixed> {
        SymMatrix3 {
            m11: fx(0x2c0000000),
            m12: fx(-0x180000000),
            m13: fx(0x90000000),
            m22: fx(0x140000000),
            m23: fx(0x1e0000000),
            m33: fx(-0x240000000),
        }
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__baseline() {
        let _s = black_box(bench_input());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__jacobi_3_sweeps() {
        let s = black_box(bench_input());
        let d = eigen_sweeps3(s);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__jacobi_4_sweeps() {
        let s = black_box(bench_input());
        let d = SymmetricEigen3Trait::new(s);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__jacobi_5_sweeps() {
        let s = black_box(bench_input());
        let d = eigen_sweeps5(s);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__jacobi_6_sweeps() {
        let s = black_box(bench_input());
        let d = eigen_sweeps6(s);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__no_renormalisation() {
        let s = black_box(bench_input());
        let j = Jacobi3Impl::<Fixed>::start(s).sweep().sweep().sweep().sweep();
        let d = finish_without_renormalisation(j);
        assert!(d.eigenvalues.x <= d.eigenvalues.y && d.eigenvalues.y <= d.eigenvalues.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_new__diagonal_input() {
        // Every rotation is skipped: the cost of the early exits alone.
        let s = black_box(SymMatrix3Trait::from_diagonal(v3i(7, -2, 1)));
        let d = SymmetricEigen3Trait::new(s);
        assert!(d.eigenvalues == v3i(-2, 1, 7));
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_eigenvalues__baseline() {
        let _s = black_box(bench_input());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_eigenvalues__without_eigenvectors() {
        let s = black_box(bench_input());
        let v = SymmetricEigen3Trait::eigenvalues(s);
        assert!(v.x <= v.y && v.y <= v.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_eigenvalues__via_new() {
        let s = black_box(bench_input());
        let v = SymmetricEigen3Trait::new(s).eigenvalues;
        assert!(v.x <= v.y && v.y <= v.z);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_sweep__baseline() {
        let _j = black_box(Jacobi3Impl::<Fixed>::start(bench_input()));
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_sweep__three_rotations() {
        let j = black_box(Jacobi3Impl::<Fixed>::start(bench_input()));
        let j = j.sweep();
        // `m12` was zeroed by the first rotation and refilled by the next two.
        assert!(j.s.m12 != Real::ZERO);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_sweep__three_rotations_without_eigenvectors() {
        let s = black_box(bench_input());
        let s = Jacobi3Impl::<Fixed>::sweep_s(s);
        assert!(s.m12 != Real::ZERO);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_recompose__baseline() {
        let _d = black_box(SymmetricEigen3Trait::new(bench_input()));
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen3_recompose__quadform() {
        let d = black_box(SymmetricEigen3Trait::new(bench_input()));
        let s = d.recompose();
        assert!(s.m11 != Real::ZERO);
    }
}
