//! `Matrix2`: a statically sized 2x2 matrix (upstream `nalgebra::Matrix2`).
//!
//! Every sum of products goes through a fused `Real` kernel: one rounding (floor) and one overflow
//! check per output scalar. Operators (`+`, `-`, unary `-`, `*` between matrices and their
//! assigning forms) are implemented in this module, so they need no import; the other operations
//! are methods of `Matrix2Trait`.

use core::ops::{AddAssign, MulAssign, SubAssign};
use simba::scalar::Real;
use super::sym_matrix2::SymMatrix2;
use super::vector2::Vector2;

/// A 2x2 matrix. `mRC` is the component at row `R`, column `C`.
///
/// Fields are declared in column-major order, so `Serde` matches upstream's storage order, while
/// `new` takes its arguments in row-major order like upstream.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Matrix2<T> {
    pub m11: T,
    pub m21: T,
    pub m12: T,
    pub m22: T,
}

/// Methods of `Matrix2<T>` for any `Real` scalar.
#[generate_trait]
pub impl Matrix2Impl<
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
> of Matrix2Trait<T> {
    // --- constructors --------------------------------------------------------------------------

    /// The matrix with the given components, in ROW-major order. Upstream: `Matrix2::new`.
    #[inline(always)]
    fn new(m11: T, m12: T, m21: T, m22: T) -> Matrix2<T> {
        Matrix2 { m11, m21, m12, m22 }
    }

    /// The zero matrix. Upstream: `Matrix2::zeros`.
    #[inline(always)]
    fn zeros() -> Matrix2<T> {
        Matrix2 { m11: R::ZERO, m21: R::ZERO, m12: R::ZERO, m22: R::ZERO }
    }

    /// The identity matrix. Upstream: `Matrix2::identity`.
    #[inline(always)]
    fn identity() -> Matrix2<T> {
        Matrix2 { m11: R::ONE, m21: R::ZERO, m12: R::ZERO, m22: R::ONE }
    }

    /// The diagonal matrix `diag(d.x, d.y)`. Upstream: `Matrix2::from_diagonal`.
    #[inline(always)]
    fn from_diagonal(d: Vector2<T>) -> Matrix2<T> {
        Matrix2 { m11: d.x, m21: R::ZERO, m12: R::ZERO, m22: d.y }
    }

    /// The matrix `e * I`. Upstream: `Matrix2::from_diagonal_element`.
    #[inline(always)]
    fn from_diagonal_element(e: T) -> Matrix2<T> {
        Matrix2 { m11: e, m21: R::ZERO, m12: R::ZERO, m22: e }
    }

    /// The matrix whose columns are `c1`, `c2`. Upstream: `Matrix2::from_columns`.
    #[inline(always)]
    fn from_columns(c1: Vector2<T>, c2: Vector2<T>) -> Matrix2<T> {
        Matrix2 { m11: c1.x, m21: c1.y, m12: c2.x, m22: c2.y }
    }

    /// The matrix whose rows are `r1`, `r2`. Upstream: `Matrix2::from_rows`.
    #[inline(always)]
    fn from_rows(r1: Vector2<T>, r2: Vector2<T>) -> Matrix2<T> {
        Matrix2 { m11: r1.x, m21: r2.x, m12: r1.y, m22: r2.y }
    }

    /// The outer product `a * bᵀ`: each component is one floored product. Panics with the
    /// scalar's overflow error. Upstream: `a * b.transpose()`.
    #[inline(always)]
    fn from_outer(a: Vector2<T>, b: Vector2<T>) -> Matrix2<T> {
        Matrix2 { m11: a.x * b.x, m21: a.y * b.x, m12: a.x * b.y, m22: a.y * b.y }
    }

    // --- accessors -----------------------------------------------------------------------------

    /// First column. Upstream: `column(0)`.
    #[inline(always)]
    fn column1(self: Matrix2<T>) -> Vector2<T> {
        Vector2 { x: self.m11, y: self.m21 }
    }

    /// Second column. Upstream: `column(1)`.
    #[inline(always)]
    fn column2(self: Matrix2<T>) -> Vector2<T> {
        Vector2 { x: self.m12, y: self.m22 }
    }

    /// First row, as a (column) vector. Upstream: `row(0).transpose()`.
    #[inline(always)]
    fn row1(self: Matrix2<T>) -> Vector2<T> {
        Vector2 { x: self.m11, y: self.m12 }
    }

    /// Second row, as a (column) vector. Upstream: `row(1).transpose()`.
    #[inline(always)]
    fn row2(self: Matrix2<T>) -> Vector2<T> {
        Vector2 { x: self.m21, y: self.m22 }
    }

    /// The diagonal `(m11, m22)`. Upstream: `diagonal`.
    #[inline(always)]
    fn diagonal(self: Matrix2<T>) -> Vector2<T> {
        Vector2 { x: self.m11, y: self.m22 }
    }

    // --- exact operations ----------------------------------------------------------------------

    /// The transpose. Exact. Upstream: `transpose`.
    #[inline(always)]
    fn transpose(self: Matrix2<T>) -> Matrix2<T> {
        Matrix2 { m11: self.m11, m21: self.m12, m12: self.m21, m22: self.m22 }
    }

    /// `m11 + m22`. Exact; panics on overflow. Upstream: `trace`.
    #[inline(always)]
    fn trace(self: Matrix2<T>) -> T {
        self.m11 + self.m22
    }

    /// Component-wise absolute value. Panics on the scalar's `MIN`. Upstream: `abs`.
    #[inline(always)]
    fn abs(self: Matrix2<T>) -> Matrix2<T> {
        Matrix2 {
            m11: R::abs(self.m11),
            m21: R::abs(self.m21),
            m12: R::abs(self.m12),
            m22: R::abs(self.m22),
        }
    }

    // --- products ------------------------------------------------------------------------------

    /// `self * k`: each component is one floored product. Panics on overflow.
    /// Upstream: `self * k`.
    #[inline(always)]
    fn scale(self: Matrix2<T>, k: T) -> Matrix2<T> {
        Matrix2 { m11: self.m11 * k, m21: self.m21 * k, m12: self.m12 * k, m22: self.m22 * k }
    }

    /// Component-wise (Hadamard) product: each component is one floored product. Panics on
    /// overflow. Upstream: `component_mul`.
    #[inline(always)]
    fn component_mul(self: Matrix2<T>, rhs: Matrix2<T>) -> Matrix2<T> {
        Matrix2 {
            m11: self.m11 * rhs.m11,
            m21: self.m21 * rhs.m21,
            m12: self.m12 * rhs.m12,
            m22: self.m22 * rhs.m22,
        }
    }

    /// `self * v`: one `sum_prod2` per component (one rounding each). Panics on overflow.
    /// Upstream: `self * v`.
    #[inline(always)]
    fn mul_vec(self: Matrix2<T>, v: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: R::sum_prod2(self.m11, v.x, self.m12, v.y),
            y: R::sum_prod2(self.m21, v.x, self.m22, v.y),
        }
    }

    /// `selfᵀ * v` without forming the transpose: one `sum_prod2` per component. Panics on
    /// overflow. Upstream: `self.tr_mul(&v)`.
    #[inline(always)]
    fn tr_mul_vec(self: Matrix2<T>, v: Vector2<T>) -> Vector2<T> {
        Vector2 {
            x: R::sum_prod2(self.m11, v.x, self.m21, v.y),
            y: R::sum_prod2(self.m12, v.x, self.m22, v.y),
        }
    }

    /// `selfᵀ * rhs`: 4 `sum_prod2`. Panics on overflow. Upstream: `tr_mul`.
    fn tr_mul(self: Matrix2<T>, rhs: Matrix2<T>) -> Matrix2<T> {
        Matrix2 {
            m11: R::sum_prod2(self.m11, rhs.m11, self.m21, rhs.m21),
            m21: R::sum_prod2(self.m12, rhs.m11, self.m22, rhs.m21),
            m12: R::sum_prod2(self.m11, rhs.m12, self.m21, rhs.m22),
            m22: R::sum_prod2(self.m12, rhs.m12, self.m22, rhs.m22),
        }
    }

    /// `self * selfᵀ` as a symmetric matrix: 3 `sum_prod2` instead of 4, bit-identical to the
    /// upper triangle of `self * self.transpose()`. Panics on overflow.
    /// Upstream: `self * self.transpose()`.
    fn mul_transpose(self: Matrix2<T>) -> SymMatrix2<T> {
        SymMatrix2 {
            m11: R::norm_squared2(self.m11, self.m12),
            m12: R::sum_prod2(self.m11, self.m21, self.m12, self.m22),
            m22: R::norm_squared2(self.m21, self.m22),
        }
    }

    // --- norms ---------------------------------------------------------------------------------

    /// Squared Frobenius norm, one rounding (`norm_squared4`). Panics on overflow.
    /// Upstream: `norm_squared`.
    #[inline(always)]
    fn norm_squared(self: Matrix2<T>) -> T {
        R::norm_squared4(self.m11, self.m21, self.m12, self.m22)
    }

    /// Frobenius norm: square root of the exact, unscaled sum of squares (one rounding, no
    /// intermediate overflow: only the result must fit). Upstream: `norm`.
    #[inline(always)]
    fn norm(self: Matrix2<T>) -> T {
        R::norm4(self.m11, self.m21, self.m12, self.m22)
    }

    // --- determinant and inverse ---------------------------------------------------------------

    /// `m11 * m22 - m12 * m21` with a single rounding (`diff_prod`): the exact floor of the true
    /// determinant. Panics on overflow. Upstream: `determinant`.
    #[inline(always)]
    fn determinant(self: Matrix2<T>) -> T {
        R::diff_prod(self.m11, self.m22, self.m12, self.m21)
    }

    /// The adjugate (transposed cofactor matrix) `[[m22, -m12], [-m21, m11]]`:
    /// `self * adjugate = determinant * I`. Exact; panics on the scalar's `MIN`.
    /// No upstream equivalent (upstream `adjoint` is the conjugate transpose).
    #[inline(always)]
    fn adjugate(self: Matrix2<T>) -> Matrix2<T> {
        Matrix2 { m11: self.m22, m21: -self.m21, m12: -self.m12, m22: self.m11 }
    }

    /// The inverse, or `None` when the matrix is singular. Upstream: `try_inverse`.
    ///
    /// Singularity criterion, like upstream: the computed determinant is EXACTLY zero (no
    /// epsilon). The zero matrix is singular. A nearly singular matrix is inverted with the
    /// precision its conditioning allows, and panics with the scalar's overflow error when a
    /// component of the inverse does not fit.
    ///
    /// Algorithm: `adjugate / determinant`, one division per component (a single reciprocal
    /// followed by multiplications is cheaper but loses the low bits of `1 / det` when
    /// `|det| >> 1`). Because a fixed-point determinant of a small matrix has few significant
    /// bits, a matrix of Frobenius norm `f <= 1` is first multiplied by the INTEGER
    /// `k = floor(2 / f)` (exact products), which gives `inverse = adjugate(k * self) * (k /
    /// det(k * self))`. Matrices with `|det| >= 1/2` skip the norm computation (`f <= 1`
    /// implies `|det| <= 1/2`). Panics with the overflow error when `0 < f <= 2^-30`.
    fn try_inverse(self: Matrix2<T>) -> Option<Matrix2<T>> {
        let det = R::diff_prod(self.m11, self.m22, self.m12, self.m21);
        if det < R::HALF && det > -R::HALF {
            let f = R::norm4(self.m11, self.m21, self.m12, self.m22);
            if f == R::ZERO {
                return None;
            }
            let k = R::floor(R::div(R::TWO, f));
            if k >= R::TWO {
                let (b11, b21, b12, b22) = (self.m11 * k, self.m21 * k, self.m12 * k, self.m22 * k);
                let det_b = R::diff_prod(b11, b22, b12, b21);
                if det_b == R::ZERO {
                    return None;
                }
                let t = R::div(k, det_b);
                return Some(
                    Matrix2 { m11: b22 * t, m21: (-b21) * t, m12: (-b12) * t, m22: b11 * t },
                );
            }
            if det == R::ZERO {
                return None;
            }
        }
        Some(
            Matrix2 {
                m11: R::div(self.m22, det),
                m21: R::div(-self.m21, det),
                m12: R::div(-self.m12, det),
                m22: R::div(self.m11, det),
            },
        )
    }

    // --- approximate equality ------------------------------------------------------------------

    /// Whether every component is within `ulps` smallest units of the identity's.
    /// Upstream: `is_identity(eps)`, with the tolerance in raw units instead of a float epsilon.
    fn is_identity(self: Matrix2<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.m11, R::ONE, ulps)
            && R::abs_diff_eq(self.m21, R::ZERO, ulps)
            && R::abs_diff_eq(self.m12, R::ZERO, ulps)
            && R::abs_diff_eq(self.m22, R::ONE, ulps)
    }

    /// Whether every component of `self` is within `ulps` smallest units of `other`'s.
    /// Upstream: `abs_diff_eq`, with the tolerance in raw units instead of a float epsilon.
    fn abs_diff_eq(self: Matrix2<T>, other: Matrix2<T>, ulps: u64) -> bool {
        R::abs_diff_eq(self.m11, other.m11, ulps)
            && R::abs_diff_eq(self.m21, other.m21, ulps)
            && R::abs_diff_eq(self.m12, other.m12, ulps)
            && R::abs_diff_eq(self.m22, other.m22, ulps)
    }
}

// --- operators -----------------------------------------------------------------------------------

/// `a + b`, component-wise. Exact; panics on overflow.
pub impl Matrix2Add<T, +Add<T>, +Copy<T>, +Drop<T>> of Add<Matrix2<T>> {
    #[inline(always)]
    fn add(lhs: Matrix2<T>, rhs: Matrix2<T>) -> Matrix2<T> {
        Matrix2 {
            m11: lhs.m11 + rhs.m11,
            m21: lhs.m21 + rhs.m21,
            m12: lhs.m12 + rhs.m12,
            m22: lhs.m22 + rhs.m22,
        }
    }
}

/// `a - b`, component-wise. Exact; panics on overflow.
pub impl Matrix2Sub<T, +Sub<T>, +Copy<T>, +Drop<T>> of Sub<Matrix2<T>> {
    #[inline(always)]
    fn sub(lhs: Matrix2<T>, rhs: Matrix2<T>) -> Matrix2<T> {
        Matrix2 {
            m11: lhs.m11 - rhs.m11,
            m21: lhs.m21 - rhs.m21,
            m12: lhs.m12 - rhs.m12,
            m22: lhs.m22 - rhs.m22,
        }
    }
}

/// `-a`, component-wise. Exact; panics on the scalar's `MIN`.
pub impl Matrix2Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<Matrix2<T>> {
    #[inline(always)]
    fn neg(a: Matrix2<T>) -> Matrix2<T> {
        Matrix2 { m11: -a.m11, m21: -a.m21, m12: -a.m12, m22: -a.m22 }
    }
}

/// `a * b` (matrix product): 4 `sum_prod2`, one rounding per component. Panics on overflow.
pub impl Matrix2Mul<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of Mul<Matrix2<T>> {
    fn mul(lhs: Matrix2<T>, rhs: Matrix2<T>) -> Matrix2<T> {
        Matrix2 {
            m11: R::sum_prod2(lhs.m11, rhs.m11, lhs.m12, rhs.m21),
            m21: R::sum_prod2(lhs.m21, rhs.m11, lhs.m22, rhs.m21),
            m12: R::sum_prod2(lhs.m11, rhs.m12, lhs.m12, rhs.m22),
            m22: R::sum_prod2(lhs.m21, rhs.m12, lhs.m22, rhs.m22),
        }
    }
}

/// `a += b`.
pub impl Matrix2AddAssign<T, +Add<T>, +Copy<T>, +Drop<T>> of AddAssign<Matrix2<T>, Matrix2<T>> {
    #[inline(always)]
    fn add_assign(ref self: Matrix2<T>, rhs: Matrix2<T>) {
        self = self + rhs;
    }
}

/// `a -= b`.
pub impl Matrix2SubAssign<T, +Sub<T>, +Copy<T>, +Drop<T>> of SubAssign<Matrix2<T>, Matrix2<T>> {
    #[inline(always)]
    fn sub_assign(ref self: Matrix2<T>, rhs: Matrix2<T>) {
        self = self - rhs;
    }
}

/// `a *= b` (matrix product, `a = a * b`).
pub impl Matrix2MulAssign<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>,
> of MulAssign<Matrix2<T>, Matrix2<T>> {
    #[inline(always)]
    fn mul_assign(ref self: Matrix2<T>, rhs: Matrix2<T>) {
        self = self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix_test_utils::{fx, int, m2, m2i, max_ulp_diff2, s2, v2i, v2t};
    use crate::base::sym_matrix2::SymMatrix2Trait;
    use crate::base::{oracle_matrix2, oracle_matrix2_inverse};
    use super::{Matrix2, Matrix2Trait};

    // --- losing candidates of the determinant / inverse study (kept as evidence) -----------------

    /// `adjugate / determinant` without the integer pre-scaling of small matrices.
    fn try_inverse_div(m: Matrix2<Fixed>) -> Option<Matrix2<Fixed>> {
        let adj = m.adjugate();
        let det = m.determinant();
        if det == Real::ZERO {
            return None;
        }
        Some(
            Matrix2 {
                m11: adj.m11 / det, m21: adj.m21 / det, m12: adj.m12 / det, m22: adj.m22 / det,
            },
        )
    }

    /// `adjugate * (1 / determinant)`: one reciprocal, 4 multiplications.
    fn try_inverse_recip(m: Matrix2<Fixed>) -> Option<Matrix2<Fixed>> {
        let det = m.determinant();
        if det == Real::ZERO {
            return None;
        }
        Some(m.adjugate().scale(det.recip()))
    }

    /// `(cases above the oracle tolerance, worst error in ulp)` of an inverse candidate.
    fn inverse_failures(variant: u8) -> (u32, u128) {
        let mut cases = oracle_matrix2_inverse::matrix2_try_inverse_cases();
        let mut failures = 0;
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let got = match variant {
                0 => m2(a).try_inverse(),
                1 => try_inverse_div(m2(a)),
                _ => try_inverse_recip(m2(a)),
            };
            let err = max_ulp_diff2(got.unwrap(), m2(expected));
            if err > tol.into() {
                failures += 1;
            }
            worst = core::cmp::max(worst, err);
        }
        (failures, worst)
    }

    // --- constructors and accessors --------------------------------------------------------------

    #[test]
    fn test_new_is_row_major() {
        let m = Matrix2Trait::new(int(1), int(2), int(3), int(4));
        assert!(m == m2i([[1, 2], [3, 4]]));
        assert!(m.m12 == int(2) && m.m21 == int(3));
    }

    #[test]
    fn test_serde_is_column_major() {
        let m = m2i([[1, 2], [3, 4]]);
        let mut out = array![];
        m.serialize(ref out);
        let one: felt252 = 0x100000000;
        assert!(out == array![1 * one, 3 * one, 2 * one, 4 * one]);
    }

    #[test]
    fn test_zeros_identity() {
        assert!(Matrix2Trait::<Fixed>::zeros() == m2i([[0, 0], [0, 0]]));
        assert!(Matrix2Trait::<Fixed>::identity() == m2i([[1, 0], [0, 1]]));
        assert!(Matrix2Trait::<Fixed>::zeros() == Default::default());
    }

    #[test]
    fn test_from_diagonal() {
        assert!(Matrix2Trait::from_diagonal(v2i(2, -3)) == m2i([[2, 0], [0, -3]]));
        assert!(Matrix2Trait::from_diagonal_element(int(7)) == m2i([[7, 0], [0, 7]]));
    }

    #[test]
    fn test_from_columns_from_rows() {
        let (r1, r2) = (v2i(1, 2), v2i(3, 4));
        assert!(Matrix2Trait::from_rows(r1, r2) == m2i([[1, 2], [3, 4]]));
        assert!(Matrix2Trait::from_columns(r1, r2) == m2i([[1, 3], [2, 4]]));
    }

    #[test]
    fn test_columns_rows_diagonal() {
        let m = m2i([[1, 2], [3, 4]]);
        assert!(m.column1() == v2i(1, 3));
        assert!(m.column2() == v2i(2, 4));
        assert!(m.row1() == v2i(1, 2));
        assert!(m.row2() == v2i(3, 4));
        assert!(m.diagonal() == v2i(1, 4));
        assert!(Matrix2Trait::from_columns(m.column1(), m.column2()) == m);
        assert!(Matrix2Trait::from_rows(m.row1(), m.row2()) == m);
    }

    // --- exact operations
    // --------------------------------------------------------------------------

    #[test]
    fn test_add_oracle() {
        let mut cases = oracle_matrix2::matrix2_add_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m2(a) + m2(b) == m2(expected));
            let mut acc = m2(a);
            acc += m2(b);
            assert!(acc == m2(expected));
        }
    }

    #[test]
    fn test_sub_oracle() {
        let mut cases = oracle_matrix2::matrix2_sub_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m2(a) - m2(b) == m2(expected));
            let mut acc = m2(a);
            acc -= m2(b);
            assert!(acc == m2(expected));
        }
    }

    #[test]
    fn test_neg() {
        let m = m2i([[1, -2], [-3, 0]]);
        assert!(-m == m2i([[-1, 2], [3, 0]]));
        assert!(m + (-m) == Matrix2Trait::zeros());
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_add_overflow_panics() {
        let m = black_box(Matrix2Trait::from_diagonal_element(Real::<Fixed>::MAX));
        let _ = m + m;
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_neg_min_panics() {
        let _ = -black_box(Matrix2Trait::from_diagonal_element(Real::<Fixed>::MIN));
    }

    #[test]
    fn test_transpose_oracle() {
        let mut cases = oracle_matrix2::matrix2_transpose_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            assert!(m2(a).transpose() == m2(expected));
            assert!(m2(a).transpose().transpose() == m2(a));
        }
    }

    #[test]
    fn test_trace_oracle() {
        let mut cases = oracle_matrix2::matrix2_trace_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            assert!(m2(a).trace() == fx(expected));
        }
    }

    #[test]
    fn test_abs() {
        let m = m2i([[1, -2], [-3, 0]]);
        assert!(m.abs() == m2i([[1, 2], [3, 0]]));
    }

    // --- products
    // ----------------------------------------------------------------------------------

    #[test]
    fn test_scale_oracle() {
        let mut cases = oracle_matrix2::matrix2_scale_cases();
        while let Some(case) = cases.pop_front() {
            let (a, k, expected, _) = *case;
            assert!(m2(a).scale(fx(k)) == m2(expected));
        }
    }

    #[test]
    fn test_component_mul_exact() {
        let a = m2i([[1, -2], [-4, 5]]);
        let b = m2i([[2, 2], [3, 3]]);
        assert!(a.component_mul(b) == m2i([[2, -4], [-12, 15]]));
        // 0.5 ulp floors to 0, -0.5 ulp to -1 ulp, 1.5 ulp to 1 ulp, -1.5 ulp to -2 ulp.
        let h = Matrix2Trait::from_diagonal_element(Real::<Fixed>::HALF);
        let e = Matrix2Trait::from_diagonal(v2t((1, -1)));
        assert!(e.component_mul(h) == Matrix2Trait::from_diagonal(v2t((0, -1))));
    }

    #[test]
    fn test_mul_oracle() {
        let mut cases = oracle_matrix2::matrix2_mul_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m2(a) * m2(b) == m2(expected));
            let mut acc = m2(a);
            acc *= m2(b);
            assert!(acc == m2(expected));
        }
    }

    #[test]
    fn test_mul_exact_and_identity() {
        let a = m2i([[1, 2], [3, 4]]);
        let b = m2i([[-1, 0], [3, 2]]);
        assert!(a * b == m2i([[5, 4], [9, 8]]));
        assert!(a * Matrix2Trait::identity() == a);
        assert!(Matrix2Trait::identity() * a == a);
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_mul_overflow_panics() {
        let m = black_box(Matrix2Trait::from_diagonal_element(int(65536)));
        let _ = m * m;
    }

    #[test]
    fn test_mul_vec_oracle() {
        let mut cases = oracle_matrix2::matrix2_mul_vec_cases();
        while let Some(case) = cases.pop_front() {
            let (a, v, expected, _) = *case;
            assert!(m2(a).mul_vec(v2t(v)) == v2t(expected));
            assert!(m2(a).transpose().tr_mul_vec(v2t(v)) == v2t(expected));
        }
    }

    #[test]
    fn test_mul_vec_tr_mul_vec_exact() {
        let a = m2i([[1, 2], [3, 4]]);
        assert!(a.mul_vec(v2i(1, 0)) == v2i(1, 3));
        assert!(a.tr_mul_vec(v2i(1, 0)) == v2i(1, 2));
    }

    #[test]
    fn test_tr_mul_oracle() {
        let mut cases = oracle_matrix2::matrix2_tr_mul_cases();
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, _) = *case;
            assert!(m2(a).tr_mul(m2(b)) == m2(expected));
            assert!(m2(a).transpose() * m2(b) == m2(expected));
        }
    }

    #[test]
    fn test_from_outer_oracle() {
        let mut cases = oracle_matrix2::matrix2_outer_cases();
        while let Some(case) = cases.pop_front() {
            let (u, v, expected, _) = *case;
            assert!(Matrix2Trait::from_outer(v2t(u), v2t(v)) == m2(expected));
        }
    }

    #[test]
    fn test_mul_transpose_matches_generic_product() {
        // small, unit and medium cases (`large` squares do not fit).
        let mut cases = oracle_matrix2::matrix2_mul_cases().slice(0, 12);
        while let Some(case) = cases.pop_front() {
            let (a, b, _, _) = *case;
            assert!(m2(a).mul_transpose().to_matrix() == m2(a) * m2(a).transpose());
            assert!(m2(b).mul_transpose().to_matrix() == m2(b) * m2(b).transpose());
        }
        let one = 0x100000000;
        assert!(m2i([[1, 2], [3, 4]]).mul_transpose() == s2((5 * one, 11 * one, 25 * one)));
    }

    // --- norms
    // -------------------------------------------------------------------------------------

    #[test]
    fn test_norm_exact() {
        let m = m2i([[2, -2], [-2, 2]]);
        assert!(m.norm_squared() == int(16));
        assert!(m.norm() == int(4));
        assert!(Matrix2Trait::<Fixed>::zeros().norm() == int(0));
        // The squared norm (2 * 2^58) does not fit, the norm does.
        assert!(Matrix2Trait::from_diagonal_element(int(0x20000000)).norm() >= int(0x20000000));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_norm_squared_overflow_panics() {
        black_box(Matrix2Trait::from_diagonal_element(int(0x20000000))).norm_squared();
    }

    // --- determinant and inverse
    // -------------------------------------------------------------------

    #[test]
    fn test_determinant_exact() {
        assert!(m2i([[1, 2], [3, 4]]).determinant() == int(-2));
        assert!(m2i([[2, 0], [0, -3]]).determinant() == int(-6));
        assert!(m2i([[1, 2], [2, 4]]).determinant() == int(0));
        assert!(Matrix2Trait::<Fixed>::identity().determinant() == int(1));
    }

    #[test]
    fn test_determinant_oracle() {
        let mut cases = oracle_matrix2_inverse::matrix2_determinant_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, _) = *case;
            assert!(m2(a).determinant() == fx(expected));
        }
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_determinant_overflow_panics() {
        black_box(Matrix2Trait::from_diagonal_element(int(65536))).determinant();
    }

    #[test]
    fn test_adjugate_identity() {
        let m = m2i([[2, -1], [5, 3]]);
        let det = m.determinant();
        assert!(det == int(11));
        assert!(m * m.adjugate() == Matrix2Trait::from_diagonal_element(det));
        assert!(m.adjugate() * m == Matrix2Trait::from_diagonal_element(det));
    }

    #[test]
    fn test_try_inverse_exact() {
        let m = m2i([[2, 1], [5, 3]]);
        let inv = m.try_inverse().unwrap();
        assert!(inv == m2i([[3, -1], [-5, 2]]));
        assert!(m * inv == Matrix2Trait::identity());
        let d = Matrix2Trait::from_diagonal(v2i(2, -4)).try_inverse().unwrap();
        assert!(d == Matrix2Trait::from_diagonal(v2t((0x80000000, -0x40000000))));
        assert!(
            Matrix2Trait::<Fixed>::identity().try_inverse().unwrap() == Matrix2Trait::identity(),
        );
    }

    #[test]
    fn test_try_inverse_oracle() {
        let mut cases = oracle_matrix2_inverse::matrix2_try_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let err = max_ulp_diff2(m2(a).try_inverse().unwrap(), m2(expected));
            assert!(err <= tol.into(), "inverse error {err} > {tol}");
        }
    }

    #[test]
    fn test_try_inverse_candidates_error() {
        // Oracle, 30 well-conditioned matrices (10 small, 10 unit, 10 medium):
        // (cases above the oracle tolerance, worst error in ulp) of the shipped algorithm, of
        // `adjugate / det` without pre-scaling and of `adjugate * (1 / det)`.
        assert!(inverse_failures(0) == (0, 8));
        assert!(inverse_failures(1) == (1, 1111));
        assert!(inverse_failures(2) == (6, 1111));
    }

    #[test]
    fn test_try_inverse_product_is_identity() {
        let mut cases = oracle_matrix2_inverse::matrix2_try_inverse_cases();
        while let Some(case) = cases.pop_front() {
            let (a, _, _) = *case;
            let inv = m2(a).try_inverse().unwrap();
            // Worst residual over the oracle: 68 ulp.
            assert!((m2(a) * inv).is_identity(68));
            assert!((inv * m2(a)).is_identity(68));
        }
    }

    #[test]
    fn test_try_inverse_singular_oracle() {
        let mut cases = oracle_matrix2_inverse::matrix2_try_inverse_singular_cases();
        while let Some(case) = cases.pop_front() {
            let (a, is_some, _) = *case;
            assert!(m2(a).try_inverse().is_some() == is_some);
        }
        assert!(Matrix2Trait::<Fixed>::zeros().try_inverse().is_none());
        assert!(m2i([[1, 2], [2, 4]]).try_inverse().is_none());
    }

    #[test]
    fn test_try_inverse_near_singular_oracle() {
        let mut cases = oracle_matrix2_inverse::matrix2_try_inverse_near_singular_cases();
        while let Some(case) = cases.pop_front() {
            let (a, expected, tol) = *case;
            let inv = m2(a).try_inverse().unwrap();
            assert!(max_ulp_diff2(inv, m2(expected)) <= tol.into());
        }
    }

    #[test]
    fn test_try_inverse_small_scale() {
        // 2^-12 * I: without pre-scaling the determinant (2^-24: 8 significant bits).
        let m = Matrix2Trait::from_diagonal_element(fx(0x100000));
        let expected = Matrix2Trait::from_diagonal_element(int(4096));
        assert!(m.determinant() == fx(0x100));
        assert!(try_inverse_div(m).unwrap() == expected);
        // Relative error below 2^-33 (1 ulp on 4096).
        let inv = m.try_inverse().unwrap();
        assert!(inv.abs_diff_eq(expected, 1));
        assert!(!inv.abs_diff_eq(expected, 0));
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_try_inverse_tiny_norm_panics() {
        let _ = black_box(Matrix2Trait::from_diagonal_element(fx(1))).try_inverse();
    }

    // --- approximate equality
    // ----------------------------------------------------------------------

    #[test]
    fn test_is_identity_abs_diff_eq() {
        let id = Matrix2Trait::<Fixed>::identity();
        let mut m = id;
        m.m21 = fx(3);
        m.m22 = fx(0x100000000 - 2);
        assert!(m.is_identity(3) && !m.is_identity(2));
        assert!(m.abs_diff_eq(id, 3) && !m.abs_diff_eq(id, 2));
        assert!(id.is_identity(0) && id.abs_diff_eq(id, 0));
    }

    // --- gas benchmarks
    // ----------------------------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_matrix2_new__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_new__struct() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        assert!(Matrix2Trait::new(a.m11, a.m12, a.m21, a.m22) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_zeros__baseline() {
        let e = black_box(m2([[0, 0], [0, 0]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_zeros__const() {
        let e = black_box(m2([[0, 0], [0, 0]]));
        assert!(Matrix2Trait::<Fixed>::zeros() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_identity__baseline() {
        let e = black_box(m2([[4294967296, 0], [0, 4294967296]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_identity__const() {
        let e = black_box(m2([[4294967296, 0], [0, 4294967296]]));
        assert!(Matrix2Trait::<Fixed>::identity() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_diagonal__baseline() {
        let _v = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(m2([[-7543252641, 0], [0, 4885438966]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_diagonal__struct() {
        let v = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(m2([[-7543252641, 0], [0, 4885438966]]));
        assert!(Matrix2Trait::from_diagonal(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_diagonal_element__baseline() {
        let _k = black_box(fx(-7516192768));
        let e = black_box(m2([[-7516192768, 0], [0, -7516192768]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_diagonal_element__struct() {
        let k = black_box(fx(-7516192768));
        let e = black_box(m2([[-7516192768, 0], [0, -7516192768]]));
        assert!(Matrix2Trait::from_diagonal_element(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_columns__baseline() {
        let _c1 = black_box(v2t((5594399379, -7444297509)));
        let _c2 = black_box(v2t((2839048663, 5944454799)));
        let e = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_columns__struct() {
        let c1 = black_box(v2t((5594399379, -7444297509)));
        let c2 = black_box(v2t((2839048663, 5944454799)));
        let e = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        assert!(Matrix2Trait::from_columns(c1, c2) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_rows__baseline() {
        let _r1 = black_box(v2t((5594399379, 2839048663)));
        let _r2 = black_box(v2t((-7444297509, 5944454799)));
        let e = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_rows__struct() {
        let r1 = black_box(v2t((5594399379, 2839048663)));
        let r2 = black_box(v2t((-7444297509, 5944454799)));
        let e = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        assert!(Matrix2Trait::from_rows(r1, r2) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_outer__baseline() {
        let _u = black_box(v2t((2347498971, -7037259012)));
        let _v = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(m2([[-4122913306, 2670232892], [12359540590, -8004740671]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_from_outer__products() {
        let u = black_box(v2t((2347498971, -7037259012)));
        let v = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(m2([[-4122913306, 2670232892], [12359540590, -8004740671]]));
        assert!(Matrix2Trait::from_outer(u, v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_column__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(v2t((2839048663, 5944454799)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_column__second() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(v2t((2839048663, 5944454799)));
        assert!(a.column2() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_row__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(v2t((-7444297509, 5944454799)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_row__second() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(v2t((-7444297509, 5944454799)));
        assert!(a.row2() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_diagonal__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(v2t((5594399379, 5944454799)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_diagonal__struct() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(v2t((5594399379, 5944454799)));
        assert!(a.diagonal() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_transpose__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[5594399379, -7444297509], [2839048663, 5944454799]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_transpose__struct() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[5594399379, -7444297509], [2839048663, 5944454799]]));
        assert!(a.transpose() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_trace__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(fx(11538854178));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_trace__sum() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(fx(11538854178));
        assert!(a.trace() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_abs__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[5594399379, 2839048663], [7444297509, 5944454799]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_abs__componentwise() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[5594399379, 2839048663], [7444297509, 5944454799]]));
        assert!(a.abs() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_add__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[9087438363, -295638616], [-4067378544, -2328483166]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_add__operator() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[9087438363, -295638616], [-4067378544, -2328483166]]));
        assert!(a + b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_add__assign() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[9087438363, -295638616], [-4067378544, -2328483166]]));
        let mut r = a;
        r += b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_sub__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[2101360395, 5973735942], [-10821216474, 14217392764]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_sub__operator() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[2101360395, 5973735942], [-10821216474, 14217392764]]));
        assert!(a - b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_sub__assign() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[2101360395, 5973735942], [-10821216474, 14217392764]]));
        let mut r = a;
        r -= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_neg__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[-5594399379, -2839048663], [7444297509, -5944454799]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_neg__operator() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[-5594399379, -2839048663], [7444297509, -5944454799]]));
        assert!(-a == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_scale__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _k = black_box(fx(-7516192768));
        let e = black_box(m2([[-9790198914, -4968335161], [13027520640, -10402795899]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_scale__products() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let k = black_box(fx(-7516192768));
        let e = black_box(m2([[-9790198914, -4968335161], [13027520640, -10402795899]]));
        assert!(a.scale(k) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_component_mul__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[4549849574, -2072083235], [-5853080526, -11450170025]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_component_mul__products() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[4549849574, -2072083235], [-5853080526, -11450170025]]));
        assert!(a.component_mul(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_mul_vec__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _v = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(v2t((-6596084900, 19836120296)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_mul_vec__fused() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let v = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(v2t((-6596084900, 19836120296)));
        assert!(a.mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_tr_mul_vec__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _v = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(v2t((-18293184465, 1775475633)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_tr_mul_vec__fused() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let v = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(v2t((-18293184465, 1775475633)));
        assert!(a.tr_mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_tr_mul_vec__transpose_mul_vec() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let v = black_box(v2t((-7543252641, 4885438966)));
        let e = black_box(v2t((-18293184465, 1775475633)));
        assert!(a.transpose().mul_vec(v) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_mul__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[6782052199, -9551636418], [-1380517907, -6016940132]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_mul__fused() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[6782052199, -9551636418], [-1380517907, -6016940132]]));
        assert!(a * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_mul__assign() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[6782052199, -9551636418], [-1380517907, -6016940132]]));
        let mut r = a;
        r *= b;
        assert!(r == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_tr_mul__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[-1303230952, 10256077842], [6982788863, -13522253260]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_tr_mul__fused() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[-1303230952, 10256077842], [6982788863, -13522253260]]));
        assert!(a.tr_mul(b) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_tr_mul__transpose_mul() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[3493038984, -3134687279], [3376918965, -8272937965]]));
        let e = black_box(m2([[-1303230952, 10256077842], [6982788863, -13522253260]]));
        assert!(a.transpose() * b == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_mul_transpose__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(s2((9163632458, -5767163102, 21130337440)));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_mul_transpose__structured() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(s2((9163632458, -5767163102, 21130337440)));
        assert!(a.mul_transpose() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_mul_transpose__generic() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(s2((9163632458, -5767163102, 21130337440)));
        assert!(SymMatrix2Trait::from_matrix_unchecked(a * a.transpose()) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_norm_squared__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(fx(30293969899));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_norm_squared__fused() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(fx(30293969899));
        assert!(a.norm_squared() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_norm__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(fx(11406647622));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_norm__fused() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(fx(11406647622));
        assert!(a.norm() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_determinant__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(fx(12663746514));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_determinant__fused() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(fx(12663746514));
        assert!(a.determinant() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_adjugate__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[5944454799, -2839048663], [7444297509, 5594399379]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_adjugate__cofactors() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(m2([[5944454799, -2839048663], [7444297509, 5594399379]]));
        assert!(a.adjugate() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_try_inverse__baseline() {
        let _a = black_box(m2([[-5146602846, 2781339837], [533542917, 1900613592]]));
        let e = black_box(m2([[-3112122077, 4554249819], [873639280, 8427202879]]));
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_try_inverse__prescaled_det_ge_half() {
        let a = black_box(m2([[-5146602846, 2781339837], [533542917, 1900613592]]));
        let e = black_box(m2([[-3112122077, 4554249819], [873639280, 8427202879]]));
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_try_inverse__prescaled_norm_gt_one() {
        let a = black_box(m2([[-4407632017, -216812900], [-2360417318, -1480186601]]));
        let e = black_box(m2([[-4541423616, 665212901], [7242097005, -13523243703]]));
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_try_inverse__prescaled_small() {
        let a = black_box(m2([[-30484050, -320085972], [-224450602, -21365614]]));
        let e = black_box(m2([[5536085856, -82938099629], [-58157832642, 7898781568]]));
        assert!(a.try_inverse().unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_try_inverse__alt_div() {
        let a = black_box(m2([[-5146602846, 2781339837], [533542917, 1900613592]]));
        let e = black_box(m2([[-3112122077, 4554249819], [873639280, 8427202879]]));
        assert!(try_inverse_div(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_try_inverse__alt_recip() {
        let a = black_box(m2([[-5146602846, 2781339837], [533542917, 1900613592]]));
        let e = black_box(m2([[-3112122077, 4554249819], [873639280, 8427202880]]));
        assert!(try_inverse_recip(a).unwrap() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_try_inverse_singular__baseline() {
        let _a = black_box(m2([[17179869184, 21474836480], [34359738368, 42949672960]]));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_try_inverse_singular__none() {
        let a = black_box(m2([[17179869184, 21474836480], [34359738368, 42949672960]]));
        let e = black_box(true);
        assert!(a.try_inverse().is_none() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_is_identity__baseline() {
        let _a = black_box(m2([[4294967296, 0], [0, 4294967296]]));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_is_identity__all_compared() {
        let a = black_box(m2([[4294967296, 0], [0, 4294967296]]));
        let e = black_box(true);
        assert!(a.is_identity(2) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_abs_diff_eq__baseline() {
        let _a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let _b = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_matrix2_abs_diff_eq__all_compared() {
        let a = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let b = black_box(m2([[5594399379, 2839048663], [-7444297509, 5944454799]]));
        let e = black_box(true);
        assert!(a.abs_diff_eq(b, 2) == e);
    }
}
