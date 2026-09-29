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

use core::internal::revoke_ap_tracking;
use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::internal::base::sym_matrix3::{SymMatrix3, SymMatrix3Trait};
use nalgebra_static3::base::matrix3::Matrix3Trait;
use nalgebra_static3::base::vector3::Vector3Trait;
use nalgebra_static3::internal::base::matrix3::Matrix3InternalTrait;
use simba::scalar::Real;
use crate::internal::linalg::symmetric_eigen3::SymmetricEigen3InternalTrait;

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
#[derive(Copy, Drop, Serde, Debug)]
pub struct SymmetricEigen3<T> {
    /// The three eigenvalues, ascending.
    pub eigenvalues: Vector3<T>,
    /// The matching unit eigenvectors, as columns (`det = +1`).
    pub eigenvectors: Matrix3<T>,
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
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of SymmetricEigen3Trait<T> {
    /// The eigendecomposition of the symmetric `m` by **four** cyclic Jacobi sweeps, unrolled.
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
    /// Like upstream, only the LOWER triangle of `m` is read (the entries at row `i`, column `j`
    /// with `i >= j`): the strictly upper triangle is ignored and the symmetry of `m` is NOT
    /// checked.
    /// Upstream: `SymmetricEigen::new` (iterative QR, different algorithm and different order).
    #[inline(always)]
    fn new(m: Matrix3<T>) -> SymmetricEigen3<T> {
        SymmetricEigen3InternalTrait::new_sym(
            SymMatrix3 { m11: m.m11, m12: m.m21, m13: m.m31, m22: m.m22, m23: m.m32, m33: m.m33 },
        )
    }

    /// `new`, or `None` when the four sweeps did not reach upstream's convergence criterion
    /// `|s_ij| <= eps * (|s_ii| + |s_jj|)` on every off-diagonal entry of the final state (one
    /// fused product pair each, floored: with `eps = 0` a leftover raw unit returns `None`).
    /// `max_niter` is accepted for signature parity and ignored: the iteration count is the
    /// constant budget of the type (upstream's `0 = unlimited` maps onto it too). Bit-identical
    /// to `new` when `Some`. Upstream: `SymmetricEigen::try_new`.
    fn try_new(m: Matrix3<T>, eps: T, max_niter: usize) -> Option<SymmetricEigen3<T>> {
        let _ = max_niter;
        SymmetricEigen3InternalTrait::try_new_sym(
            SymMatrix3 { m11: m.m11, m12: m.m21, m13: m.m31, m22: m.m22, m23: m.m32, m33: m.m33 },
            eps,
        )
    }

    /// `V * diag(eigenvalues) * Vᵀ`, the symmetric matrix the decomposition came from, up to the
    /// rounding of the decomposition. Only the 6 independent components are computed (a
    /// structured quadratic form), then mirrored. Panics on overflow. Upstream:
    /// `SymmetricEigen::recompose`.
    #[inline(always)]
    fn recompose(self: SymmetricEigen3<T>) -> Matrix3<T> {
        SymMatrix3Trait::quadform(self.eigenvectors, self.eigenvalues).to_matrix()
    }
}

/// Upstream's `SquareMatrix` methods that go through the symmetric eigen decomposition, on
/// `Matrix3`. Import `Matrix3SymmetricEigenTrait` to use them.
#[generate_trait]
pub impl Matrix3SymmetricEigenImpl<
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
> of Matrix3SymmetricEigenTrait<T> {
    /// The eigendecomposition of the symmetric `self` (lower triangle read), see
    /// `SymmetricEigen3Trait::new`. Upstream: `Matrix::symmetric_eigen`.
    #[inline(always)]
    fn symmetric_eigen(self: Matrix3<T>) -> SymmetricEigen3<T> {
        SymmetricEigen3Trait::new(self)
    }

    /// See `SymmetricEigen3Trait::try_new`. Upstream: `Matrix::try_symmetric_eigen`.
    #[inline(always)]
    fn try_symmetric_eigen(
        self: Matrix3<T>, eps: T, max_niter: usize,
    ) -> Option<SymmetricEigen3<T>> {
        SymmetricEigen3Trait::try_new(self, eps, max_niter)
    }

    /// The eigenvalues of the symmetric `self` alone (lower triangle read), ascending, without the
    /// eigenvectors: the same four sweeps as `new` with the accumulation of the
    /// rotation dropped, bit-identical to `new(self).eigenvalues` and 45 % cheaper. Upstream:
    /// `Matrix::symmetric_eigenvalues`.
    #[inline(always)]
    fn symmetric_eigenvalues(self: Matrix3<T>) -> Vector3<T> {
        let m = self;
        SymmetricEigen3InternalTrait::eigenvalues(
            SymMatrix3 { m11: m.m11, m12: m.m21, m13: m.m31, m22: m.m22, m23: m.m32, m33: m.m33 },
        )
    }
}
