// Specialisations of `Matrix4`, spliced verbatim by shapegen.py (format: `library.py`).
// @item Matrix4Kernels struct
/// Internal kernels of `Matrix4`.
#[generate_trait]
impl Matrix4Kernels<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of Matrix4KernelsTrait<T> {
    /// `a*b - c*d + e*f`, accumulated exactly and rounded once.
    #[inline(always)]
    fn pmp(a: T, b: T, c: T, d: T, e: T, f: T) -> T {
        let w = R::wide_add_prod(R::wide_zero(), a, b);
        R::wide_rescale(R::wide_add_prod(R::wide_sub_prod(w, c, d), e, f))
    }

    /// `-a*b + c*d - e*f`, accumulated exactly and rounded once.
    #[inline(always)]
    fn mpm(a: T, b: T, c: T, d: T, e: T, f: T) -> T {
        let w = R::wide_sub_prod(R::wide_zero(), a, b);
        R::wide_rescale(R::wide_sub_prod(R::wide_add_prod(w, c, d), e, f))
    }

    /// `(adjugate, determinant)` from the twelve 2x2 minors of the two upper rows (`s0..s5`)
    /// and of the two lower rows (`c0..c5`): 12 `diff_prod`, then 16 three-term and one six-term
    /// exact accumulations (two roundings per output).
    #[inline(always)]
    fn adjugate_determinant(m: Matrix4<T>) -> (Matrix4<T>, T) {
        let s0 = R::diff_prod(m.m11, m.m22, m.m21, m.m12);
        let s1 = R::diff_prod(m.m11, m.m23, m.m21, m.m13);
        let s2 = R::diff_prod(m.m11, m.m24, m.m21, m.m14);
        let s3 = R::diff_prod(m.m12, m.m23, m.m22, m.m13);
        let s4 = R::diff_prod(m.m12, m.m24, m.m22, m.m14);
        let s5 = R::diff_prod(m.m13, m.m24, m.m23, m.m14);
        let c5 = R::diff_prod(m.m33, m.m44, m.m43, m.m34);
        let c4 = R::diff_prod(m.m32, m.m44, m.m42, m.m34);
        let c3 = R::diff_prod(m.m32, m.m43, m.m42, m.m33);
        let c2 = R::diff_prod(m.m31, m.m44, m.m41, m.m34);
        let c1 = R::diff_prod(m.m31, m.m43, m.m41, m.m33);
        let c0 = R::diff_prod(m.m31, m.m42, m.m41, m.m32);
        let w = R::wide_add_prod(R::wide_zero(), s0, c5);
        let w = R::wide_sub_prod(w, s1, c4);
        let w = R::wide_add_prod(w, s2, c3);
        let w = R::wide_add_prod(w, s3, c2);
        let w = R::wide_sub_prod(w, s4, c1);
        let det = R::wide_rescale(R::wide_add_prod(w, s5, c0));
        let adj = Matrix4 {
            m11: Self::pmp(m.m22, c5, m.m23, c4, m.m24, c3),
            m21: Self::mpm(m.m21, c5, m.m23, c2, m.m24, c1),
            m31: Self::pmp(m.m21, c4, m.m22, c2, m.m24, c0),
            m41: Self::mpm(m.m21, c3, m.m22, c1, m.m23, c0),
            m12: Self::mpm(m.m12, c5, m.m13, c4, m.m14, c3),
            m22: Self::pmp(m.m11, c5, m.m13, c2, m.m14, c1),
            m32: Self::mpm(m.m11, c4, m.m12, c2, m.m14, c0),
            m42: Self::pmp(m.m11, c3, m.m12, c1, m.m13, c0),
            m13: Self::pmp(m.m42, s5, m.m43, s4, m.m44, s3),
            m23: Self::mpm(m.m41, s5, m.m43, s2, m.m44, s1),
            m33: Self::pmp(m.m41, s4, m.m42, s2, m.m44, s0),
            m43: Self::mpm(m.m41, s3, m.m42, s1, m.m43, s0),
            m14: Self::mpm(m.m32, s5, m.m33, s4, m.m34, s3),
            m24: Self::pmp(m.m31, s5, m.m33, s2, m.m34, s1),
            m34: Self::mpm(m.m31, s4, m.m32, s2, m.m34, s0),
            m44: Self::pmp(m.m31, s3, m.m32, s1, m.m33, s0),
        };
        (adj, det)
    }
}
// @method determinant
/// The determinant, from the 2x2 minors of the two upper rows (`s0..s5`) and of the two lower
/// rows (`c0..c5`): `s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0`, i.e. 12 `diff_prod` (one
/// rounding each) and one exact six-term accumulation (second rounding). 25 % cheaper and one
/// rounding less than the cofactor expansion along a row. Panics on overflow.
/// Upstream: `determinant`.
fn determinant(self: Matrix4<T>) -> T {
    let s0 = R::diff_prod(self.m11, self.m22, self.m21, self.m12);
    let s1 = R::diff_prod(self.m11, self.m23, self.m21, self.m13);
    let s2 = R::diff_prod(self.m11, self.m24, self.m21, self.m14);
    let s3 = R::diff_prod(self.m12, self.m23, self.m22, self.m13);
    let s4 = R::diff_prod(self.m12, self.m24, self.m22, self.m14);
    let s5 = R::diff_prod(self.m13, self.m24, self.m23, self.m14);
    let c5 = R::diff_prod(self.m33, self.m44, self.m43, self.m34);
    let c4 = R::diff_prod(self.m32, self.m44, self.m42, self.m34);
    let c3 = R::diff_prod(self.m32, self.m43, self.m42, self.m33);
    let c2 = R::diff_prod(self.m31, self.m44, self.m41, self.m34);
    let c1 = R::diff_prod(self.m31, self.m43, self.m41, self.m33);
    let c0 = R::diff_prod(self.m31, self.m42, self.m41, self.m32);
    let w = R::wide_add_prod(R::wide_zero(), s0, c5);
    let w = R::wide_sub_prod(w, s1, c4);
    let w = R::wide_add_prod(w, s2, c3);
    let w = R::wide_add_prod(w, s3, c2);
    let w = R::wide_sub_prod(w, s4, c1);
    R::wide_rescale(R::wide_add_prod(w, s5, c0))
}
// @method try_inverse
/// The inverse, or `None` when the matrix is singular. Upstream: `try_inverse`.
///
/// Singularity criterion, like upstream: the computed determinant is EXACTLY zero (no
/// epsilon). The zero matrix is singular. A nearly singular matrix is inverted with the
/// precision its conditioning allows, and panics with the scalar's overflow error when a
/// component of the inverse does not fit.
///
/// Algorithm: `adjugate / determinant` (see `determinant`), one division per component (a
/// single reciprocal followed by 16 multiplications is cheaper but loses the low bits of
/// `1 / det` when `|det| >> 1`). Because a fixed-point determinant of a small matrix has few
/// significant bits, a matrix of Frobenius norm `f <= 1` is first multiplied by the INTEGER
/// `k = floor(2 / f)` (exact products), which gives `inverse = adjugate(k * self) * (k /
/// det(k * self))`. Matrices with `|det| >= 1/2` skip the norm computation (`f <= 1`
/// implies `|det| <= 1/16`). Panics with the overflow error when `0 < f <= 2^-30`.
///
/// Re-ranked on `fixed` 0.3.0 (WP 7.2): the unscaled branch divides through ONE prepared
/// divisor (`Real::div16`, bit-identical to per-element division). `adjugate / det` without the
/// pre-scaling costs 161 670 gas through `Real::div16` (`bench_matrix4_try_inverse__alt_div_n`)
/// against 196 820, and `adjugate * (1 / det)` — upstream's 4x4 formula (MESA's inverse) —
/// is cheaper still, but they leave respectively 6 and 10 of the 30 oracle cases outside their
/// tolerance (`test_try_inverse_candidates_error`), so the pre-scaled algorithm stays. The
/// charged gas is that of the costliest branch (the pre-scaled one): the three
/// `bench_matrix4_try_inverse__prescaled_*` benchmarks measure the same figure.
fn try_inverse(self: Matrix4<T>) -> Option<Matrix4<T>> {
    let (adj, det) = Matrix4Kernels::adjugate_determinant(self);
    if det < R::HALF && det > -R::HALF {
        let f = Self::norm(self);
        if f == R::zero() {
            return None;
        }
        let k = R::floor(R::div(R::TWO, f));
        if k >= R::TWO {
            let (adj_b, det_b) = Matrix4Kernels::adjugate_determinant(Self::scale(self, k));
            if det_b == R::zero() {
                return None;
            }
            return Some(Self::scale(adj_b, R::div(k, det_b)));
        }
        if det == R::zero() {
            return None;
        }
    }
    let (m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44) =
        R::div16(
        adj.m11,
        adj.m21,
        adj.m31,
        adj.m41,
        adj.m12,
        adj.m22,
        adj.m32,
        adj.m42,
        adj.m13,
        adj.m23,
        adj.m33,
        adj.m43,
        adj.m14,
        adj.m24,
        adj.m34,
        adj.m44,
        det,
    );
    Some(
        Matrix4 {
            m11, m21, m31, m41, m12, m22, m32, m42, m13, m23, m33, m43, m14, m24, m34, m44,
        },
    )
}
// @doc internal
/// Crate-internal kernels of `Matrix4<T>` with no upstream method of that name or shape (WP 8.0:
/// the public API is strictly upstream's). The structured kernels of DESIGN D4 (`from_outer`,
/// `adjugate`) and the unrolled row / column accessors that stand for upstream `row(i)` /
/// `column(i)` views, used by the decompositions.
// @internal adjugate
/// The adjugate (transposed cofactor matrix): `self * adjugate = determinant * I`. 12
/// `diff_prod` (2x2 minors) then 16 exact three-term accumulations: two roundings per
/// component. Panics on overflow.
/// No upstream equivalent (upstream `adjoint` is the conjugate transpose).
fn adjugate(self: Matrix4<T>) -> Matrix4<T> {
    let (adj, _) = Matrix4Kernels::adjugate_determinant(self);
    adj
}
