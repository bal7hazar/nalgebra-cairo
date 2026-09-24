// Specialisations of `Matrix2`, spliced verbatim by shapegen.py (format: `library.py`).
// @use super::sym_matrix2::SymMatrix2
// @method determinant
/// `m11 * m22 - m12 * m21` with a single rounding (`diff_prod`): the exact floor of the true
/// determinant. Panics on overflow. Upstream: `determinant`.
#[inline(always)]
fn determinant(self: Matrix2<T>) -> T {
    R::diff_prod(self.m11, self.m22, self.m12, self.m21)
}
// @method try_inverse
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
///
/// Re-ranked on `fixed` 0.3.0 (WP 7.2): the unscaled branch divides through ONE prepared
/// divisor (`Real::div4`, bit-identical to per-element division). `adjugate / det` without the
/// pre-scaling — upstream's 2x2 formula — costs 21 570 gas through `Real::div4`
/// (`bench_matrix2_try_inverse__alt_div_n`) against 33 970, and `adjugate * (1 / det)` is
/// cheaper still, but they leave respectively 1 and 4 of the 30 oracle cases outside their
/// tolerance (`test_try_inverse_candidates_error`), so the pre-scaled algorithm stays. The
/// charged gas is that of the costliest branch (the pre-scaled one): the three
/// `bench_matrix2_try_inverse__prescaled_*` benchmarks measure the same figure.
fn try_inverse(self: Matrix2<T>) -> Option<Matrix2<T>> {
    let det = R::diff_prod(self.m11, self.m22, self.m12, self.m21);
    if det < R::HALF && det > -R::HALF {
        let f = R::norm4(self.m11, self.m21, self.m12, self.m22);
        if f == R::zero() {
            return None;
        }
        let k = R::floor(R::div(R::TWO, f));
        if k >= R::TWO {
            let (b11, b21, b12, b22) = (self.m11 * k, self.m21 * k, self.m12 * k, self.m22 * k);
            let det_b = R::diff_prod(b11, b22, b12, b21);
            if det_b == R::zero() {
                return None;
            }
            let t = R::div(k, det_b);
            return Some(
                Matrix2 { m11: b22 * t, m21: (-b21) * t, m12: (-b12) * t, m22: b11 * t },
            );
        }
        if det == R::zero() {
            return None;
        }
    }
    let (m11, m21, m12, m22) = R::div4(self.m22, -self.m21, -self.m12, self.m11, det);
    Some(Matrix2 { m11, m21, m12, m22 })
}
// @doc internal
/// Crate-internal kernels of `Matrix2<T>` with no upstream method of that name or shape (WP 8.0:
/// the public API is strictly upstream's). The structured kernels of DESIGN D4 (`from_outer`,
/// `adjugate`, `mul_transpose` into a `SymMatrix2`) and the unrolled column accessors that stand
/// for upstream `column(i)` views, used by the decompositions.
// @internal mul_transpose
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
// @internal adjugate
/// The adjugate (transposed cofactor matrix) `[[m22, -m12], [-m21, m11]]`:
/// `self * adjugate = determinant * I`. Exact; panics on the scalar's `MIN`.
/// No upstream equivalent (upstream `adjoint` is the conjugate transpose).
#[inline(always)]
fn adjugate(self: Matrix2<T>) -> Matrix2<T> {
    Matrix2 { m11: self.m22, m21: -self.m21, m12: -self.m12, m22: self.m11 }
}
