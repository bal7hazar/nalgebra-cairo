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
#[inline(always)]
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
///
/// WP 13-OPT-3, same bits and same panics as before: `k = floor(2 / f) >= 2` exactly when
/// `f <= 1` (`2 / f` is rounded to nearest: one ulp above 1 it is already 2 ulp below 2), so the
/// division is made on the pre-scaled branch only; the six cofactors the determinant does not use
/// are computed after that branch, which never needs them (its components are at most 1 in
/// magnitude, so they could not overflow there), and before the singularity test, as before.
#[inline(always)]
fn try_inverse(self: Matrix3<T>) -> Option<Matrix3<T>> {
    let a11 = R::diff_prod(self.m22, self.m33, self.m23, self.m32);
    let a21 = R::diff_prod(self.m23, self.m31, self.m21, self.m33);
    let a31 = R::diff_prod(self.m21, self.m32, self.m22, self.m31);
    let det = R::sum_prod3(self.m11, a11, self.m12, a21, self.m13, a31);
    let (a12, a22, a32, a13, a23, a33) = if det < R::HALF && det > -R::HALF {
        let f = Self::norm(self);
        if f == R::zero() {
            return None;
        }
        if f <= R::one() {
            let k = R::floor(R::div(R::TWO, f));
            let b = Self::scale(self, k);
            let adj_b = Matrix3InternalTrait::adjugate(b);
            let det_b = R::sum_prod3(b.m11, adj_b.m11, b.m12, adj_b.m21, b.m13, adj_b.m31);
            if det_b == R::zero() {
                return None;
            }
            return Some(Self::scale(adj_b, R::div(k, det_b)));
        }
        let rest = (
        R::diff_prod(self.m13, self.m32, self.m12, self.m33),
        R::diff_prod(self.m11, self.m33, self.m13, self.m31),
        R::diff_prod(self.m12, self.m31, self.m11, self.m32),
        R::diff_prod(self.m12, self.m23, self.m13, self.m22),
        R::diff_prod(self.m13, self.m21, self.m11, self.m23),
        R::diff_prod(self.m11, self.m22, self.m12, self.m21),
        );
        if det == R::zero() {
            return None;
        }
        rest
    } else {
        (
        R::diff_prod(self.m13, self.m32, self.m12, self.m33),
        R::diff_prod(self.m11, self.m33, self.m13, self.m31),
        R::diff_prod(self.m12, self.m31, self.m11, self.m32),
        R::diff_prod(self.m12, self.m23, self.m13, self.m22),
        R::diff_prod(self.m13, self.m21, self.m11, self.m23),
        R::diff_prod(self.m11, self.m22, self.m12, self.m21),
        )
    };
    let (m11, m21, m31, m12, m22, m32, m13, m23, m33) = R::div9(
        a11, a21, a31, a12, a22, a32, a13, a23, a33, det,
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
// @item try_inverse_reference end
/// `try_inverse` before WP 13-OPT-3 against the current one, bit for bit (results and panics).
#[cfg(test)]
mod try_inverse_reference {
    use fixed::Fixed;
    use simba::scalar::FixedReal as R;
    use super::{Matrix3, Matrix3Trait};

    /// The internal `adjugate` (`#[inline(always)]`), written in place: naming the crate-internal
    /// trait here would make the package split import it outside the tests.
    fn adjugate(self: Matrix3<Fixed>) -> Matrix3<Fixed> {
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

    /// `Matrix3Trait::try_inverse` before WP 13-OPT-3 (the same body, called, not inlined): the
    /// whole adjugate first, then `k = floor(2 / f)` on every small determinant.
    fn try_inverse_reference(self: Matrix3<Fixed>) -> Option<Matrix3<Fixed>> {
        let adj = adjugate(self);
        let det = R::sum_prod3(self.m11, adj.m11, self.m12, adj.m21, self.m13, adj.m31);
        if det < R::HALF && det > -R::HALF {
            let f = Matrix3Trait::norm(self);
            if f == R::zero() {
                return None;
            }
            let k = R::floor(R::div(R::TWO, f));
            if k >= R::TWO {
                let b = Matrix3Trait::scale(self, k);
                let adj_b = adjugate(b);
                let det_b = R::sum_prod3(b.m11, adj_b.m11, b.m12, adj_b.m21, b.m13, adj_b.m31);
                if det_b == R::zero() {
                    return None;
                }
                return Some(Matrix3Trait::scale(adj_b, R::div(k, det_b)));
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

    fn fx(raw: i64) -> Fixed {
        Fixed { raw }
    }

    /// The matrix of row-major raw components `r`.
    fn m(r: [i64; 9]) -> Matrix3<Fixed> {
        let [a, b, c, d, e, f, g, h, i] = r;
        Matrix3 {
            m11: fx(a),
            m12: fx(b),
            m13: fx(c),
            m21: fx(d),
            m22: fx(e),
            m23: fx(f),
            m31: fx(g),
            m32: fx(h),
            m33: fx(i),
        }
    }

    /// Deterministic 64-bit LCG (Knuth's MMIX constants).
    fn next(ref state: u128) -> u128 {
        state = (state * 6364136223846793005 + 1442695040888963407) % 0x10000000000000000;
        state
    }

    /// A raw value uniform in `[-bound, bound]`.
    fn draw(ref state: u128, bound: u128) -> i64 {
        let r: i128 = (next(ref state) % (2 * bound + 1)).try_into().unwrap();
        let b: i128 = bound.try_into().unwrap();
        (r - b).try_into().unwrap()
    }

    /// A raw value of magnitude uniform in `[lo, lo + span]` and a random sign.
    fn draw_mag(ref state: u128, lo: u128, span: u128) -> i64 {
        let v: i64 = (lo + next(ref state) % (span + 1)).try_into().unwrap();
        if next(ref state) % 2 == 0 {
            v
        } else {
            -v
        }
    }

    /// The branch of `try_inverse` a matrix takes: 0 `|det| >= 1/2`, 1 pre-scaled (`f <= 1`),
    /// 2 small determinant with `f > 1` (or the zero matrix).
    fn branch(a: Matrix3<Fixed>) -> u32 {
        let det = Matrix3Trait::determinant(a);
        if !(det < R::HALF && det > -R::HALF) {
            0
        } else if Matrix3Trait::norm(a) <= R::one() && Matrix3Trait::norm(a) != R::zero() {
            1
        } else {
            2
        }
    }

    /// Edge cases (zero, identity, singular, the pre-scaling threshold `f = 1` and one ulp above
    /// it, 1-ulp and extreme components; `f > 2^-30`, below which both bodies panic) and four deterministic bands of 50 matrices: diagonally
    /// dominant with diagonals from 2 to 2^10 (`|det| >= 1/2`), components up to 1/3 (pre-scaled,
    /// `f <= 1`), nearly singular rows `r3 = r1 + r2 + δ` with components from 1 to 4 (small
    /// determinant, `f > 1`), and one large diagonal component up to 2^20 with off-diagonals up to
    /// 1/2.
    /// Every branch is taken at least 40 times.
    #[test]
    fn test_try_inverse_matches_reference() {
        let one: i64 = 0x100000000;
        let half: i64 = 0x80000000;
        let mut cases: Array<Matrix3<Fixed>> = array![
            m([0, 0, 0, 0, 0, 0, 0, 0, 0]), m([one, 0, 0, 0, one, 0, 0, 0, one]),
            m([-one, 0, 0, 0, -one, 0, 0, 0, -one]), m([one, one, one, one, one, one, one, one, one]),
            m([one, 2 * one, 3 * one, 4 * one, 5 * one, 6 * one, 7 * one, 8 * one, 9 * one]),
            m([16, 0, 0, 0, 16, 0, 0, 0, 16]), m([-16, 1, 0, 1, 16, -1, 0, 1, 16]),
            m([half, half, 0, 0, half, 0, 0, 0, half]), m([half, 0, 0, 0, half, 0, 0, 0, half]),
            m([one + 1, 0, 0, 0, 1, 0, 0, 0, 1]), m([one, 0, 0, 0, 1, 0, 0, 0, 1]),
            m([one - 1, 0, 0, 0, 1, 0, 0, 0, 1]), m([0, one, 0, one, 0, 0, 0, 0, one]),
            m([0x7fffffff, 0, 0, 0, one, 0, 0, 0, one]),
            m([0x40000000000000, 0, 0, 0, one, 0, 0, 0, one]),
            m([-0x40000000000000, 1, -1, 1, one, 0, -1, 0, 0x100000]),
            m([3 * half, 0, 0, 0, 3 * half, 0, 0, 0, 3 * half]),
            m([half, 0, 0, 0, half, 0, 0, 0, half + 1]),
            m([half, half, 0, half, half, 0, 0, 0, half]), m([16, -16, 16, -16, 16, 16, 16, 16, -16]),
            m([1651849619, 3942926787, -4111385247, -2706174340, -1356311774, -3161153896,
              1713407532, -2388220103, 1097729905]),
        ];
        let mut state: u128 = 0x13579bdf2468ace0;
        let mut i: u32 = 0;
        while i < 50 {
            let d1 = draw_mag(ref state, 2 * 0x100000000, 0x3fe00000000);
            let d2 = draw_mag(ref state, 2 * 0x100000000, 0x3fe00000000);
            let d3 = draw_mag(ref state, 2 * 0x100000000, 0x3fe00000000);
            let o = 0x100000000;
            cases
                .append(
                    m(
                        [
                            d1, draw(ref state, o), draw(ref state, o), draw(ref state, o), d2,
                            draw(ref state, o), draw(ref state, o), draw(ref state, o), d3,
                        ],
                    ),
                );
            let t = 0x55555555;
            cases
                .append(
                    m(
                        [
                            draw(ref state, t), draw(ref state, t), draw(ref state, t),
                            draw(ref state, t), draw(ref state, t), draw(ref state, t),
                            draw(ref state, t), draw(ref state, t), draw(ref state, t),
                        ],
                    ),
                );
            let (a1, a2, a3) = (
                draw_mag(ref state, 0x100000000, 0x300000000),
                draw_mag(ref state, 0x100000000, 0x300000000),
                draw_mag(ref state, 0x100000000, 0x300000000),
            );
            let (b1, b2, b3) = (
                draw_mag(ref state, 0x100000000, 0x300000000),
                draw_mag(ref state, 0x100000000, 0x300000000),
                draw_mag(ref state, 0x100000000, 0x300000000),
            );
            let dlt = 0x1000000;
            cases
                .append(
                    m(
                        [
                            a1, a2, a3, b1, b2, b3, a1 + b1 + draw(ref state, dlt),
                            a2 + b2 + draw(ref state, dlt), a3 + b3 + draw(ref state, dlt),
                        ],
                    ),
                );
            let (h, q) = (0x80000000, 0x20000000);
            cases
                .append(
                    m(
                        [
                            draw_mag(ref state, 0x100000000, 0xfffff00000000), draw(ref state, h),
                            draw(ref state, h), draw(ref state, h),
                            draw_mag(ref state, 0x80000000, 0x380000000), draw(ref state, q),
                            draw(ref state, h), draw(ref state, q),
                            draw_mag(ref state, 0x80000000, 0x380000000),
                        ],
                    ),
                );
            i += 1;
        }
        let (mut b0, mut b1, mut b2, mut n) = (0_u32, 0_u32, 0_u32, 0_u32);
        for a in cases {
            assert!(Matrix3Trait::try_inverse(a) == try_inverse_reference(a), "case {}", n);
            match branch(a) {
                0 => b0 += 1,
                1 => b1 += 1,
                _ => b2 += 1,
            }
            n += 1;
        }
        assert!(n >= 200, "{} cases", n);
        assert!(b0 >= 40 && b1 >= 40 && b2 >= 40, "branches {} {} {}", b0, b1, b2);
    }

    /// A small determinant with `f > 1` and a cofactor that overflows (`m33` of the adjugate,
    /// `2^20 * 2^20`): the new body computes it after the norm and still panics, with the same
    /// error as the reference.
    fn overflowing() -> Matrix3<Fixed> {
        m([0x10000000000000, 0, 0, 0, 0x10000000000000, 0, 0, 0, 0])
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_cofactor_overflow_panics() {
        let _ = Matrix3Trait::try_inverse(overflowing());
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_reference_cofactor_overflow_panics() {
        let _ = try_inverse_reference(overflowing());
    }
}
