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

use simba::scalar::Real;
use crate::base::matrix2::{Matrix2, Matrix2Trait};
use crate::base::sym_matrix2::{SymMatrix2, SymMatrix2Trait};
use crate::base::vector2::{Vector2, Vector2Trait};

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
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
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
    /// The eigendecomposition of `s`, in closed form.
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
    /// overflow), the eigenvector is one truncated division per component.
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
    /// Upstream: `SymmetricEigen::new`.
    fn new(s: SymMatrix2<T>) -> SymmetricEigen2<T> {
        let mean = R::sum_prod2(s.m11, R::HALF, s.m22, R::HALF);
        let d = R::diff_prod(s.m11, R::HALF, s.m22, R::HALF);
        let r = R::norm2(d, s.m12);
        let eigenvalues = Vector2 { x: mean - r, y: mean + r };
        if s.m12 == R::ZERO && d == R::ZERO {
            // Isotropic: every direction is an eigenvector, and both rows of `S - λ₁ I` vanish.
            return SymmetricEigen2 { eigenvalues, eigenvectors: Matrix2Trait::identity() };
        }
        // Rows of `S - λ₁ I`: `(d + r, m12)` and `(m12, r - d)`. Both `d + r` and `r - d` are
        // non-negative; `d > 0` makes the first one the larger, `d <= 0` the second one.
        let u = if d > R::ZERO {
            Vector2 { x: -s.m12, y: d + r }
        } else {
            Vector2 { x: d - r, y: s.m12 }
        };
        let c1 = Self::canonical_sign(u.normalize());
        SymmetricEigen2 {
            eigenvalues, eigenvectors: Matrix2 { m11: c1.x, m21: c1.y, m12: -c1.y, m22: c1.x },
        }
    }

    /// The eigendecomposition of `m`, **assumed symmetric**: only the upper triangle
    /// (`m11`, `m12`, `m22`) is read, `m21` is ignored. Upstream: `Matrix2::symmetric_eigen`
    /// (which likewise reads a single triangle).
    #[inline(always)]
    fn from_matrix(m: Matrix2<T>) -> SymmetricEigen2<T> {
        Self::new(SymMatrix2Trait::from_matrix_unchecked(m))
    }

    /// The eigenvalues of `s` alone, ascending, without the eigenvectors (half the cost: no
    /// normalisation). Upstream: `Matrix2::symmetric_eigenvalues`.
    #[inline(always)]
    fn eigenvalues(s: SymMatrix2<T>) -> Vector2<T> {
        let mean = R::sum_prod2(s.m11, R::HALF, s.m22, R::HALF);
        let r = R::norm2(R::diff_prod(s.m11, R::HALF, s.m22, R::HALF), s.m12);
        Vector2 { x: mean - r, y: mean + r }
    }

    /// `v` with the sign that makes its component of largest absolute value positive (ties go to
    /// `x`); `v` unchanged when it is zero. This is the sign convention of `eigenvectors`,
    /// documented on the struct. No upstream equivalent.
    #[inline(always)]
    fn canonical_sign(v: Vector2<T>) -> Vector2<T> {
        let dominant = if v.x.abs() >= v.y.abs() {
            v.x
        } else {
            v.y
        };
        if dominant.is_negative() {
            Vector2 { x: -v.x, y: -v.y }
        } else {
            v
        }
    }

    /// `V * diag(eigenvalues) * Vᵀ`, the symmetric matrix the decomposition came from, up to the
    /// rounding of the decomposition (measured: **4 ulp per unit of `max |m_ij|`** on the generic
    /// oracle vectors, 15 on the near-isotropic SPD ones, see `new`). Goes through
    /// `SymMatrix2::quadform`, so only the 3 independent components are computed. Panics on
    /// overflow. Upstream: `SymmetricEigen::recompose` (which returns the full matrix, see
    /// `recompose_matrix`).
    #[inline(always)]
    fn recompose(self: SymmetricEigen2<T>) -> SymMatrix2<T> {
        SymMatrix2Trait::quadform(self.eigenvectors, self.eigenvalues)
    }

    /// `recompose()` as a full `Matrix2`, bit-identical to it by symmetry.
    /// Upstream: `SymmetricEigen::recompose`.
    #[inline(always)]
    fn recompose_matrix(self: SymmetricEigen2<T>) -> Matrix2<T> {
        Self::recompose(self).to_matrix()
    }
}

#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix2::{Matrix2, Matrix2Trait};
    use crate::base::matrix_test_utils::{
        amax_s2, fx, int, max_ulp_diff_s2, max_ulp_diff_v2, s2i, s2r, ulp_diff, v2i, v2t,
    };
    use crate::base::sym_matrix2::{SymMatrix2, SymMatrix2Trait};
    use crate::base::vector2::{Vector2, Vector2Trait};
    use crate::linalg::oracle_symmetric_eigen;
    use super::{SymmetricEigen2, SymmetricEigen2Trait};

    /// `|<c_i, c_j> - delta_ij|` over the two columns, in raw units.
    fn orthonormality_error(v: Matrix2<Fixed>) -> u128 {
        let (c1, c2) = (v.column1(), v.column2());
        let mut e = ulp_diff(c1.norm(), Real::ONE);
        e = core::cmp::max(e, ulp_diff(c2.norm(), Real::ONE));
        core::cmp::max(e, ulp_diff(c1.dot(c2), Real::ZERO))
    }

    /// `|S * c_i - lambda_i * c_i|` over the two columns, in raw units.
    fn residual_error(s: SymMatrix2<Fixed>, e: SymmetricEigen2<Fixed>) -> u128 {
        let (c1, c2) = (e.eigenvectors.column1(), e.eigenvectors.column2());
        let r1 = s.mul_vec(c1) - c1.scale(e.eigenvalues.x);
        let r2 = s.mul_vec(c2) - c2.scale(e.eigenvalues.y);
        core::cmp::max(
            max_ulp_diff_v2(r1, Vector2Trait::zeros()), max_ulp_diff_v2(r2, Vector2Trait::zeros()),
        )
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_diagonal_is_exact() {
        // Already ordered: columns are the axes, in order.
        let e = SymmetricEigen2Trait::new(s2i((2, 0, 7)));
        assert!(e.eigenvalues == v2i(2, 7));
        assert!(e.eigenvectors == Matrix2Trait::identity());
        // Reversed: the permutation that sorts ascending must keep `det = +1`.
        let e = SymmetricEigen2Trait::new(s2i((7, 0, 2)));
        assert!(e.eigenvalues == v2i(2, 7));
        assert!(e.eigenvectors == Matrix2Trait::new(int(0), int(-1), int(1), int(0)));
        assert!(e.eigenvectors.determinant() == Real::ONE);
        assert!(e.recompose() == s2i((7, 0, 2)));
    }

    #[test]
    fn test_new_isotropic_is_exact() {
        let e = SymmetricEigen2Trait::new(s2i((3, 0, 3)));
        assert!(e.eigenvalues == v2i(3, 3));
        assert!(e.eigenvectors == Matrix2Trait::identity());
        assert!(e.recompose() == s2i((3, 0, 3)));
        // Zero is isotropic too, and must not divide by zero.
        let e = SymmetricEigen2Trait::new(s2i((0, 0, 0)));
        assert!(e.eigenvalues == v2i(0, 0));
        assert!(e.eigenvectors == Matrix2Trait::identity());
    }

    #[test]
    fn test_new_anti_diagonal_is_exact() {
        // [[0, 1], [1, 0]]: eigenvalues -1, 1 for (1, -1)/sqrt(2), (1, 1)/sqrt(2).
        let e = SymmetricEigen2Trait::new(s2i((0, 1, 0)));
        assert!(e.eigenvalues == v2i(-1, 1));
        // `1/sqrt(2)` reached through a floored `norm2` and a truncated division: 2 ulp of the
        // rounded constant.
        let h = Real::<Fixed>::FRAC_1_SQRT_2;
        assert!(max_ulp_diff_v2(e.eigenvectors.column1(), Vector2 { x: h, y: -h }) <= 2);
        assert!(max_ulp_diff_v2(e.eigenvectors.column2(), Vector2 { x: h, y: h }) <= 2);
        assert!(orthonormality_error(e.eigenvectors) <= 2);
    }

    #[test]
    fn test_new_integer_case_is_exact() {
        // [[5, 2], [2, 2]]: mean 3.5, d 1.5, r = sqrt(2.25 + 4) = 2.5 -> eigenvalues 1 and 6.
        let e = SymmetricEigen2Trait::new(s2i((5, 2, 2)));
        assert!(e.eigenvalues == v2i(1, 6));
        // The eigenvectors are (-1, 2)/sqrt(5) and (2, 1)/sqrt(5), irrational: 4 ulp of residual.
        assert!(residual_error(s2i((5, 2, 2)), e) <= 4);
    }

    #[test]
    fn test_new_is_deterministic_under_a_sign_flip_of_the_off_diagonal() {
        // `S` and its conjugate by diag(-1, 1) share the eigenvalues, and the sign convention is
        // on the dominant component (here `y`), so the columns mirror in `x` and the result is
        // reproducible rather than merely "some" basis. Floor rounding is not sign-symmetric
        // (`floor(-x) != -floor(x)`), so the mirrored component is only equal to 1 ulp.
        let a = SymmetricEigen2Trait::new(s2i((5, 2, 2)));
        let b = SymmetricEigen2Trait::new(s2i((5, -2, 2)));
        assert!(a.eigenvalues == b.eigenvalues);
        assert!(ulp_diff(a.eigenvectors.column1().x, -b.eigenvectors.column1().x) <= 1);
        assert!(a.eigenvectors.column1().y == b.eigenvectors.column1().y);
    }

    #[test]
    fn test_canonical_sign_picks_the_dominant_component() {
        assert!(SymmetricEigen2Trait::canonical_sign(v2i(-3, 1)) == v2i(3, -1));
        assert!(SymmetricEigen2Trait::canonical_sign(v2i(1, -3)) == v2i(-1, 3));
        // Tie (|x| == |y|): `x` decides.
        assert!(SymmetricEigen2Trait::canonical_sign(v2i(-1, 1)) == v2i(1, -1));
        assert!(SymmetricEigen2Trait::canonical_sign(v2i(0, -1)) == v2i(0, 1));
        assert!(SymmetricEigen2Trait::<Fixed>::canonical_sign(v2i(0, 0)) == v2i(0, 0));
    }

    #[test]
    fn test_from_matrix_reads_the_upper_triangle() {
        let m = Matrix2Trait::new(int(5), int(2), int(-9), int(2));
        assert!(SymmetricEigen2Trait::from_matrix(m).eigenvalues == v2i(1, 6));
    }

    #[test]
    fn test_eigenvalues_matches_new() {
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s2r(a);
            assert!(
                SymmetricEigen2Trait::eigenvalues(s) == SymmetricEigen2Trait::new(s).eigenvalues,
            );
        }
    }

    // --- oracle --------------------------------------------------------------------------------

    #[test]
    fn test_new_eigenvalues_oracle() {
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = SymmetricEigen2Trait::new(s2r(a)).eigenvalues;
            let e = max_ulp_diff_v2(got, v2t(expected));
            assert!(e <= tol.into(), "eigenvalues off by more than the oracle tolerance");
            worst = core::cmp::max(worst, e);
        }
        // Measured: the closed form is within 1 ulp of the floored exact eigenvalues.
        assert!(worst <= 1, "worst case regressed");
    }

    #[test]
    fn test_new_eigenvalues_oracle_spd() {
        let mut worst: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_spd_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = SymmetricEigen2Trait::new(s2r(a)).eigenvalues;
            let e = max_ulp_diff_v2(got, v2t(expected));
            assert!(e <= tol.into(), "eigenvalues off by more than the oracle tolerance");
            worst = core::cmp::max(worst, e);
        }
        assert!(worst <= 1, "worst case regressed");
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle() {
        let mut worst_rec: u128 = 0;
        let mut worst_orth: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s2r(a);
            let e = SymmetricEigen2Trait::new(s);
            // `det = +1` only up to the rounding of the normalised column.
            assert!(ulp_diff(e.eigenvectors.determinant(), Real::ONE) <= 4);
            let orth = orthonormality_error(e.eigenvectors);
            assert!(orth <= 8, "columns are not orthonormal");
            let rec = max_ulp_diff_s2(e.recompose(), s) / amax_s2(s);
            assert!(rec <= 8, "reconstruction is off");
            assert!(residual_error(s, e) <= 4 * amax_s2(s));
            assert!(e.recompose_matrix() == e.recompose().to_matrix());
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        // Measured worst cases over the 30 vectors.
        assert!(worst_rec <= 4 && worst_orth <= 2, "worst case regressed");
    }

    #[test]
    fn test_new_recompose_and_orthonormality_oracle_spd() {
        // SPD vectors reach much closer to isotropy than the generic ones, and the eigenvector
        // `(d - r, m12)` is then a tiny vector whose direction is coarsely quantised: the unit
        // columns are a factor 20 less accurate here than on the generic vectors. The eigenvalues
        // are not affected (still 1 ulp), and neither is the residual `S c - lambda c`.
        let mut worst_rec: u128 = 0;
        let mut worst_orth: u128 = 0;
        let mut cases = oracle_symmetric_eigen::symmetric_eigen2_eigenvalues_spd_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _expected, _tol) = *case;
            let s = s2r(a);
            let e = SymmetricEigen2Trait::new(s);
            assert!(ulp_diff(e.eigenvectors.determinant(), Real::ONE) <= 128);
            let orth = orthonormality_error(e.eigenvectors);
            assert!(orth <= 64, "columns are not orthonormal");
            let rec = max_ulp_diff_s2(e.recompose(), s) / amax_s2(s);
            assert!(rec <= 32, "reconstruction is off");
            assert!(residual_error(s, e) <= 4 * amax_s2(s));
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        assert!(worst_rec <= 15 && worst_orth <= 38, "worst case regressed");
    }

    // --- overflow ------------------------------------------------------------------------------

    #[test]
    #[should_panic(expected: 'i64_add Overflow')]
    fn test_new_overflow_panics() {
        // `mean + r` leaves the representable range.
        let s = black_box(SymMatrix2 { m11: Real::<Fixed>::MAX, m12: Real::MAX, m22: Real::MAX });
        SymmetricEigen2Trait::new(s);
    }

    // --- gas -----------------------------------------------------------------------------------

    fn bench_input() -> SymMatrix2<Fixed> {
        SymMatrix2 { m11: fx(0x2c0000000), m12: fx(-0x180000000), m22: fx(0x140000000) }
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_new__baseline() {
        let _s = black_box(bench_input());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_new__closed_form() {
        let s = black_box(bench_input());
        let d = SymmetricEigen2Trait::new(s);
        assert!(d.eigenvalues.x < d.eigenvalues.y);
        assert!(d.eigenvectors.m11 != Real::ZERO);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_eigenvalues__baseline() {
        let _s = black_box(bench_input());
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_eigenvalues__closed_form() {
        let s = black_box(bench_input());
        let v = SymmetricEigen2Trait::eigenvalues(s);
        assert!(v.x < v.y);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_recompose__baseline() {
        let _d = black_box(SymmetricEigen2Trait::new(bench_input()));
        let e = black_box(fx(0x100000000));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_symmetric_eigen2_recompose__quadform() {
        let d = black_box(SymmetricEigen2Trait::new(bench_input()));
        let s = d.recompose();
        assert!(s.m11 != Real::ZERO);
    }
}
