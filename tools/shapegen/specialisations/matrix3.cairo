// Specialisations of `Matrix3`, spliced verbatim by shapegen.py (format: `library.py`).
// @use super::sym_matrix3::SymMatrix3
// @method cross_matrix
/// The skew-symmetric matrix `[v]×` such that `[v]× * u = v × u`. Exact; panics on the
/// scalar's `MIN`.
/// Upstream: `Vector3::cross_matrix`.
#[inline(always)]
fn cross_matrix(v: Vector3<T>) -> Matrix3<T> {
    Matrix3 {
        m11: R::zero(),
        m21: v.z,
        m31: -v.y,
        m12: -v.z,
        m22: R::zero(),
        m32: v.x,
        m13: v.y,
        m23: -v.x,
        m33: R::zero(),
    }
}
// @method determinant
/// The determinant, by cofactor expansion along the first row: 3 `diff_prod` (2x2 minors,
/// one rounding each) then one `sum_prod3` (second rounding). The error against the exact
/// floor is at most `|m11| + |m12| + |m13| + 1` ulp (values, not raw: 4 ulp for a rotation).
/// Panics on overflow. Upstream: `determinant`.
fn determinant(self: Matrix3<T>) -> T {
    R::sum_prod3(
        self.m11,
        R::diff_prod(self.m22, self.m33, self.m23, self.m32),
        self.m12,
        R::diff_prod(self.m23, self.m31, self.m21, self.m33),
        self.m13,
        R::diff_prod(self.m21, self.m32, self.m22, self.m31),
    )
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
/// followed by 9 multiplications is 15 % cheaper but loses the low bits of `1 / det` when
/// `|det| >> 1`: up to 900 ulp on the oracle's `medium` matrices, against 1). Because a
/// fixed-point determinant of a small matrix has few significant bits (a triple product of
/// 0.05 is 2^-13), a matrix of Frobenius norm `f <= 1` is first multiplied by the INTEGER
/// `k = floor(2 / f)` (exact products), which gives `inverse = adjugate(k * self) * (k /
/// det(k * self))`: 27 ulp instead of 84 960 on the oracle's `small` matrices. Matrices
/// with `|det| >= 1/2` skip the norm computation (`f <= 1` implies `|det| < 1/2`). Panics
/// with the overflow error when `0 < f <= 2^-30`.
///
/// Re-ranked on `fixed` 0.3.0 (WP 7.2): the unscaled branch divides through ONE prepared
/// divisor (`Real::div9`, bit-identical to per-element division). `adjugate / det` without the
/// pre-scaling — upstream's 3x3 formula — costs 66 860 gas through `Real::div9`
/// (`bench_matrix3_try_inverse__alt_div_n`) against 88 360, and `adjugate * (1 / det)` is
/// cheaper still, but they leave respectively 2 and 9 of the 30 oracle cases outside their
/// tolerance (`test_try_inverse_candidates_error`), so the pre-scaled algorithm stays. The
/// charged gas is that of the costliest branch (the pre-scaled one): the three
/// `bench_matrix3_try_inverse__prescaled_*` benchmarks measure the same figure.
fn try_inverse(self: Matrix3<T>) -> Option<Matrix3<T>> {
    let adj = Matrix3InternalTrait::adjugate(self);
    let det = R::sum_prod3(self.m11, adj.m11, self.m12, adj.m21, self.m13, adj.m31);
    if det < R::HALF && det > -R::HALF {
        let f = Self::norm(self);
        if f == R::zero() {
            return None;
        }
        let k = R::floor(R::div(R::TWO, f));
        if k >= R::TWO {
            let b = Self::scale(self, k);
            let adj_b = Matrix3InternalTrait::adjugate(b);
            let det_b = R::sum_prod3(b.m11, adj_b.m11, b.m12, adj_b.m21, b.m13, adj_b.m31);
            if det_b == R::zero() {
                return None;
            }
            return Some(Self::scale(adj_b, R::div(k, det_b)));
        }
        if det == R::zero() {
            return None;
        }
    }
    let (m11, m21, m31, m12, m22, m32, m13, m23, m33) = R::div9(
        adj.m11, adj.m21, adj.m31, adj.m12, adj.m22, adj.m32, adj.m13, adj.m23, adj.m33, det,
    );
    Some(Matrix3 { m11, m21, m31, m12, m22, m32, m13, m23, m33 })
}
// @doc internal
/// Crate-internal kernels of `Matrix3<T>` with no upstream method of that name or shape (WP 8.0:
/// the public API is strictly upstream's). The structured kernels of DESIGN D4 (`from_outer`,
/// `cross_matrix_mul` = `v.cross_matrix() * m` without the skew matrix, `adjugate`, `mul_transpose`
/// into a `SymMatrix3`) and the unrolled row / column accessors that stand for upstream `row(i)` /
/// `column(i)` views, used by the decompositions.
// @internal cross_matrix_mul
/// `[v]× * m` without materialising `[v]×`: column `j` is `v × column_j(m)`, 9 `diff_prod`
/// (one rounding per component), bit-identical to `cross_matrix(v) * m`. Panics on overflow.
/// Upstream: `v.cross_matrix() * m`; rapier: `gcross_matrix`.
fn cross_matrix_mul(v: Vector3<T>, m: Matrix3<T>) -> Matrix3<T> {
    Matrix3 {
        m11: R::diff_prod(v.y, m.m31, v.z, m.m21),
        m21: R::diff_prod(v.z, m.m11, v.x, m.m31),
        m31: R::diff_prod(v.x, m.m21, v.y, m.m11),
        m12: R::diff_prod(v.y, m.m32, v.z, m.m22),
        m22: R::diff_prod(v.z, m.m12, v.x, m.m32),
        m32: R::diff_prod(v.x, m.m22, v.y, m.m12),
        m13: R::diff_prod(v.y, m.m33, v.z, m.m23),
        m23: R::diff_prod(v.z, m.m13, v.x, m.m33),
        m33: R::diff_prod(v.x, m.m23, v.y, m.m13),
    }
}
// @internal mul_transpose
/// `self * selfᵀ` as a symmetric matrix: 6 fused kernels instead of 9, bit-identical to the
/// upper triangle of `self * self.transpose()`. Panics on overflow.
/// Upstream: `self * self.transpose()`.
fn mul_transpose(self: Matrix3<T>) -> SymMatrix3<T> {
    SymMatrix3 {
        m11: R::norm_squared3(self.m11, self.m12, self.m13),
        m12: R::sum_prod3(self.m11, self.m21, self.m12, self.m22, self.m13, self.m23),
        m13: R::sum_prod3(self.m11, self.m31, self.m12, self.m32, self.m13, self.m33),
        m22: R::norm_squared3(self.m21, self.m22, self.m23),
        m23: R::sum_prod3(self.m21, self.m31, self.m22, self.m32, self.m23, self.m33),
        m33: R::norm_squared3(self.m31, self.m32, self.m33),
    }
}
// @internal adjugate
/// The adjugate (transposed cofactor matrix): `self * adjugate = determinant * I`. 9
/// `diff_prod`, one rounding per component. Panics on overflow.
/// No upstream equivalent (upstream `adjoint` is the conjugate transpose).
#[inline(always)]
fn adjugate(self: Matrix3<T>) -> Matrix3<T> {
    Matrix3 {
        m11: R::diff_prod(self.m22, self.m33, self.m23, self.m32),
        m21: R::diff_prod(self.m23, self.m31, self.m21, self.m33),
        m31: R::diff_prod(self.m21, self.m32, self.m22, self.m31),
        m12: R::diff_prod(self.m13, self.m32, self.m12, self.m33),
        m22: R::diff_prod(self.m11, self.m33, self.m13, self.m31),
        m32: R::diff_prod(self.m12, self.m31, self.m11, self.m32),
        m13: R::diff_prod(self.m12, self.m23, self.m13, self.m22),
        m23: R::diff_prod(self.m13, self.m21, self.m11, self.m23),
        m33: R::diff_prod(self.m11, self.m22, self.m12, self.m21),
    }
}
