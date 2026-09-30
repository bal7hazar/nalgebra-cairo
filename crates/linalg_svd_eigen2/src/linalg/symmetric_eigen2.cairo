//! `SymmetricEigen2`: eigenvalues and eigenvectors of a symmetric 2x2 matrix, in closed form.
//!
//! Upstream `nalgebra::linalg::SymmetricEigen` tridiagonalises and then runs an implicitly shifted
//! symmetric QR with Wilkinson shifts until `|off| <= eps * (|d_m| + |d_n|)` — an unbounded loop
//! with a tolerance parameter, which a proof system cannot price. In 2x2 the whole problem is a
//! quadratic (upstream itself special-cases the trailing 2x2 block in closed form,
//! `symmetric_eigen.rs`), so this port solves it directly: constant cost, no tolerance, no
//! iteration count (DESIGN D6).
//!
//! `glamx::SymmetricEigen2` (what parry / rapier use in Rust) solves the same quadratic; the
//! difference here is the choice of the eigenvector row, made on the sign of `(m11 - m22) / 2` so
//! that no cancellation can occur, and the fixed-point kernels (one rounding per output scalar).

use nalgebra_static2::internal::base::sym_matrix2::SymMatrix2Trait;
use nalgebra_types2::base::matrix2::Matrix2;
use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types2::internal::base::sym_matrix2::SymMatrix2;
use simba::scalar::Real;
use crate::internal::linalg::symmetric_eigen2::SymmetricEigen2InternalTrait;

/// The eigendecomposition `S = V * diag(eigenvalues) * Vᵀ` of a symmetric 2x2 matrix.
///
/// `eigenvalues` are sorted **ascending** (like `tools/oracle`, unlike upstream, which leaves the
/// order to the QR sweep); the columns of `eigenvectors` are the matching unit eigenvectors, in
/// the same order. `eigenvectors` is a rotation: `det(eigenvectors) = +1` by construction.
///
/// Sign convention (upstream has none; eigenvectors are only defined up to a sign):
/// **column 1 is oriented so that its component of largest absolute value is positive** (ties, i.e.
/// `|x| == |y|`, go to `x`), and column 2 is its direct perpendicular `(-y, x)`. The decomposition
/// of a given matrix is therefore a deterministic function of its raw components, as required by
/// the numeric contract of AGENTS.md.
///
/// Upstream: `SymmetricEigen { eigenvalues: OVector, eigenvectors: OMatrix }`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct SymmetricEigen2<T> {
    /// The two eigenvalues, ascending.
    pub eigenvalues: Vector2<T>,
    /// The matching unit eigenvectors, as columns (`det = +1`).
    pub eigenvectors: Matrix2<T>,
}

/// Methods of `SymmetricEigen2<T>` for any `Real` scalar.
#[generate_trait]
pub impl SymmetricEigen2Impl<
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
> of SymmetricEigen2Trait<T> {
    /// The eigendecomposition of the symmetric `m`, in closed form.
    ///
    /// With `mean = (m11 + m22) / 2`, `d = (m11 - m22) / 2` and `r = sqrt(d² + m12²)` the
    /// eigenvalues are `mean - r <= mean + r`. The eigenvector of the *smaller* eigenvalue is
    /// read off the row of `S - λ₁ I` of larger magnitude — `(d + r, m12)` when `d > 0`, else
    /// `(m12, r - d)` — which is exactly the classical cancellation-avoiding choice; the second
    /// column is its perpendicular, so the matrix is a rotation by construction and costs one
    /// normalisation instead of two.
    ///
    /// Cost: constant (no loop, no tolerance). Rounding: `mean` and `d` are one floored halving
    /// each, `r` is a floored `norm2` (the sum of squares is accumulated unscaled, so it cannot
    /// overflow), the eigenvector is one correctly rounded division per component.
    ///
    /// **Measured** bit-exactly on the 60 vectors of the `symmetric_eigen_svd` oracle suite
    /// (`small` / `unit` / `medium`, generic and SPD): eigenvalues within **1 ulp** of the floored
    /// exact result, i.e. at most 3.6 % of the oracle tolerance; `S c - λ c` within
    /// **4 ulp per unit of `max |m_ij|`**. The unit columns split by distribution: on the generic
    /// vectors they are orthonormal within **2 ulp** and `recompose()` is within **4 ulp per unit
    /// of `max |m_ij|`**; on the SPD ones, which come much closer to isotropy, the row the
    /// eigenvector is read off is itself *tiny*, so its direction is coarsely quantised and the
    /// figures are **38 ulp** and **15 ulp per unit of `max |m_ij|`**.
    ///
    /// Exact when `m11 + m22` and `m11 - m22` are even in raw units (in particular for every
    /// integer-valued matrix) and `d² + m12²` is a perfect square — diagonal and isotropic
    /// matrices included.
    ///
    /// Panics on overflow of `mean ± r` (`|m11| + |m22|` beyond the scalar's range).
    /// Like upstream, only the LOWER triangle of `m` is read (the entries at row `i`, column `j`
    /// with `i >= j`): the strictly upper triangle is ignored and the symmetry of `m` is NOT
    /// checked.
    /// Upstream: `SymmetricEigen::new`.
    #[inline(always)]
    fn new(m: Matrix2<T>) -> SymmetricEigen2<T> {
        SymmetricEigen2InternalTrait::new_sym(SymMatrix2 { m11: m.m11, m12: m.m21, m22: m.m22 })
    }

    /// `Some(new(m))`: the 2x2 decomposition is a closed form, there is nothing to converge.
    /// `eps` and `max_niter` are accepted for signature parity and ignored. Upstream:
    /// `SymmetricEigen::try_new`.
    #[inline(always)]
    fn try_new(m: Matrix2<T>, eps: T, max_niter: usize) -> Option<SymmetricEigen2<T>> {
        let _ = eps;
        let _ = max_niter;
        Some(Self::new(m))
    }

    /// `V * diag(eigenvalues) * Vᵀ`, the symmetric matrix the decomposition came from, up to the
    /// rounding of the decomposition. Only the 3 independent components are computed (a
    /// structured quadratic form), then mirrored. Panics on overflow. Upstream:
    /// `SymmetricEigen::recompose`.
    #[inline(always)]
    fn recompose(self: SymmetricEigen2<T>) -> Matrix2<T> {
        SymMatrix2Trait::quadform(self.eigenvectors, self.eigenvalues).to_matrix()
    }
}

/// Upstream's `SquareMatrix` methods that go through the symmetric eigen decomposition, on
/// `Matrix2`. Import `Matrix2SymmetricEigenTrait` to use them.
#[generate_trait]
pub impl Matrix2SymmetricEigenImpl<
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
> of Matrix2SymmetricEigenTrait<T> {
    /// The eigendecomposition of the symmetric `self` (lower triangle read), see
    /// `SymmetricEigen2Trait::new`. Upstream: `Matrix::symmetric_eigen`.
    #[inline(always)]
    fn symmetric_eigen(self: Matrix2<T>) -> SymmetricEigen2<T> {
        SymmetricEigen2Trait::new(self)
    }

    /// See `SymmetricEigen2Trait::try_new` (always `Some`). Upstream:
    /// `Matrix::try_symmetric_eigen`.
    #[inline(always)]
    fn try_symmetric_eigen(
        self: Matrix2<T>, eps: T, max_niter: usize,
    ) -> Option<SymmetricEigen2<T>> {
        SymmetricEigen2Trait::try_new(self, eps, max_niter)
    }

    /// The eigenvalues of the symmetric `self` alone (lower triangle read), ascending, without the
    /// eigenvectors (half the cost: no normalisation). Upstream:
    /// `Matrix::symmetric_eigenvalues`.
    #[inline(always)]
    fn symmetric_eigenvalues(self: Matrix2<T>) -> Vector2<T> {
        let m = self;
        SymmetricEigen2InternalTrait::eigenvalues(SymMatrix2 { m11: m.m11, m12: m.m21, m22: m.m22 })
    }
}

/// The Wilkinson shift: the eigenvalue of the symmetric 2x2 matrix `[[tmm, tmn], [tmn, tnn]]`
/// closest to `tnn`, i.e. `tnn - sgn(d) tmn² / (|d| + sqrt(d² + tmn²))` with `d = (tmm - tnn) /
/// 2` (`sgn(0) = +1`, like upstream's `signum`), and `tnn` when `tmn` is exactly zero.
///
/// Computed scale-free: `d` is one floored halving (`diff_prod` with `1/2`), the root is a
/// floored `norm2` (the squares are accumulated unscaled, no intermediate overflow), and the
/// correction is `tmn * (tmn / den)` — a correctly rounded quotient bounded by 1, then one
/// floored product — instead of upstream's `tmn² / den`, whose square underflows or overflows
/// in fixed point. Four roundings. Panics on overflow of the result. Upstream:
/// `nalgebra::linalg::wilkinson_shift`.
pub fn wilkinson_shift<
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
>(
    tmm: T, tnn: T, tmn: T,
) -> T {
    if tmn == R::zero() {
        return tnn;
    }
    let d = R::diff_prod(tmm, R::HALF, tnn, R::HALF);
    let q = tmn * R::div(tmn, d.abs() + R::norm2(d, tmn));
    if d.is_sign_negative() {
        tnn + q
    } else {
        tnn - q
    }
}
