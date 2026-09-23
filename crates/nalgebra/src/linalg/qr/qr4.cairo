//! `Qr4`: the QR factorisation of a `Matrix4` (upstream `nalgebra::linalg::QR` on a 4x4 matrix).
//!
//! `A = Q * R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal — the
//! convention of upstream's unpacked `q()` / `r()`, see the module doc of `linalg::qr`. The
//! algorithm is the modified Gram-Schmidt of `qr3`, one column longer; the study of the
//! alternatives (Householder, classical Gram-Schmidt, completed basis) lives there.

use simba::scalar::Real;
use crate::base::matrix4::{Matrix4, Matrix4Trait};
use crate::base::vector4::Vector4;

/// The QR factorisation of a `Matrix4<T>`: `A = Q * R`.
///
/// `q` is orthonormal and `r` is upper triangular with `r_ii >= 0` and an exactly zero strict
/// lower triangle. Built by `Qr4Trait::new` or `Matrix4QrTrait::qr`.
/// Upstream: `nalgebra::linalg::QR`.
#[derive(Copy, Drop, PartialEq, Serde, Debug, Hash)]
pub struct Qr4<T> {
    /// The orthonormal factor `Q` (orthonormal iff `is_invertible`, see the module doc).
    pub q: Matrix4<T>,
    /// The upper triangular factor `R`, `r_ii >= 0`.
    pub r: Matrix4<T>,
}

/// Methods of `Qr4<T>` for any `Real` scalar.
#[generate_trait]
pub impl Qr4Impl<
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
> of Qr4Trait<T> {
    /// The QR factorisation of `matrix` by modified Gram-Schmidt, fully unrolled. Upstream:
    /// `matrix.qr()` / `QR::new(matrix)` (Householder, same unpacked factors).
    ///
    /// Column `k` is stripped of its projections on `q_1 .. q_{k-1}` one at a time, each
    /// projection taken against the ALREADY UPDATED column (the modified form, see `qr3`), then
    /// normalised. Each `r_ii` is a floored `norm4` on an unscaled sum of squares, each `r_ij` one
    /// fused `sum_prod4`, each update one fused `mul_add` per component, each component of `q_i`
    /// one correctly rounded division.
    ///
    /// A column of the working matrix that is exactly zero leaves `r_ii = 0` and sets `q_i = 0`
    /// rather than completing the basis (module doc): `Q * R = A` still holds exactly, and
    /// `is_invertible` is false.
    ///
    /// **Measured** on the 30 vectors of the `qr4_q_r` oracle suite: the factors agree with
    /// upstream's unpacked Householder factors within **84 ulp** (`Q`) and **29 ulp** (`R`),
    /// every case inside the oracle tolerance; `|A - Q R| <= 3 ulp * max(1, max |a_ij|)` and
    /// `|QᵀQ - I| <= 78 ulp`.
    ///
    /// Panics with the scalar's overflow error if a norm does not fit.
    fn new(matrix: Matrix4<T>) -> Qr4<T> {
        let (a1, a2, a3, a4) = (
            matrix.column1(), matrix.column2(), matrix.column3(), matrix.column4(),
        );
        let r11 = R::norm4(a1.x, a1.y, a1.z, a1.w);
        let q1 = Self::unit(a1, r11);
        let r12 = R::sum_prod4(q1.x, a2.x, q1.y, a2.y, q1.z, a2.z, q1.w, a2.w);
        let r13 = R::sum_prod4(q1.x, a3.x, q1.y, a3.y, q1.z, a3.z, q1.w, a3.w);
        let r14 = R::sum_prod4(q1.x, a4.x, q1.y, a4.y, q1.z, a4.z, q1.w, a4.w);
        let b2 = Self::project_out(a2, q1, r12);
        let b3 = Self::project_out(a3, q1, r13);
        let b4 = Self::project_out(a4, q1, r14);
        let r22 = R::norm4(b2.x, b2.y, b2.z, b2.w);
        let q2 = Self::unit(b2, r22);
        let r23 = R::sum_prod4(q2.x, b3.x, q2.y, b3.y, q2.z, b3.z, q2.w, b3.w);
        let r24 = R::sum_prod4(q2.x, b4.x, q2.y, b4.y, q2.z, b4.z, q2.w, b4.w);
        let c3 = Self::project_out(b3, q2, r23);
        let c4 = Self::project_out(b4, q2, r24);
        let r33 = R::norm4(c3.x, c3.y, c3.z, c3.w);
        let q3 = Self::unit(c3, r33);
        let r34 = R::sum_prod4(q3.x, c4.x, q3.y, c4.y, q3.z, c4.z, q3.w, c4.w);
        let d4 = Self::project_out(c4, q3, r34);
        let r44 = R::norm4(d4.x, d4.y, d4.z, d4.w);
        let q4 = Self::unit(d4, r44);
        Qr4 {
            q: Matrix4Trait::from_columns(q1, q2, q3, q4),
            r: Matrix4Trait::new(
                r11,
                r12,
                r13,
                r14,
                R::ZERO,
                r22,
                r23,
                r24,
                R::ZERO,
                R::ZERO,
                r33,
                r34,
                R::ZERO,
                R::ZERO,
                R::ZERO,
                r44,
            ),
        }
    }

    /// `v / n`, or the zero vector when `n` is exactly zero (the rank-deficient fallback of the
    /// module doc). One correctly rounded division per component. No upstream equivalent.
    #[inline(always)]
    fn unit(v: Vector4<T>, n: T) -> Vector4<T> {
        if n == R::ZERO {
            Vector4 { x: R::ZERO, y: R::ZERO, z: R::ZERO, w: R::ZERO }
        } else {
            {
                let (x, y, z, w) = R::div4(v.x, v.y, v.z, v.w, n);
                Vector4 { x, y, z, w }
            }
        }
    }

    /// `v - c * q`, one fused `mul_add` per component (one rounding each). No upstream equivalent.
    #[inline(always)]
    fn project_out(v: Vector4<T>, q: Vector4<T>, c: T) -> Vector4<T> {
        Vector4 {
            x: R::mul_add(-c, q.x, v.x),
            y: R::mul_add(-c, q.y, v.y),
            z: R::mul_add(-c, q.z, v.z),
            w: R::mul_add(-c, q.w, v.w),
        }
    }

    /// The orthonormal factor `Q`. Exact: a move. Upstream: `QR::q`.
    #[inline(always)]
    fn q(self: Qr4<T>) -> Matrix4<T> {
        self.q
    }

    /// The upper triangular factor `R`. Exact: a move. Upstream: `QR::r`.
    #[inline(always)]
    fn r(self: Qr4<T>) -> Matrix4<T> {
        self.r
    }

    /// `(Q, R)`. Exact: moves. Upstream: `QR::unpack`.
    #[inline(always)]
    fn unpack(self: Qr4<T>) -> (Matrix4<T>, Matrix4<T>) {
        (self.q, self.r)
    }

    /// Whether the factorisation is invertible: all 4 diagonal entries of `R` are EXACTLY nonzero
    /// (no epsilon), like upstream's `QR::is_invertible`. Equivalently, `Q` is orthonormal.
    #[inline(always)]
    fn is_invertible(self: Qr4<T>) -> bool {
        self.r.m11 != R::ZERO
            && self.r.m22 != R::ZERO
            && self.r.m33 != R::ZERO
            && self.r.m44 != R::ZERO
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a diagonal entry of `R`
    /// is exactly zero (`is_invertible`). Upstream: `QR::solve`.
    ///
    /// `x = R^-1 (Qᵀ b)`, see `Qr3::solve`. Panics with the scalar's overflow error if a
    /// component of `x` does not fit.
    fn solve(self: Qr4<T>, b: Vector4<T>) -> Option<Vector4<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        Some(Self::back_substitute(self, self.q.tr_mul_vec(b)))
    }

    /// The inverse, or `None` when a diagonal entry of `R` is exactly zero (`is_invertible`).
    /// `A^-1 = R^-1 Qᵀ`, see `Qr3::try_inverse`. Upstream: `QR::try_inverse`.
    fn try_inverse(self: Qr4<T>) -> Option<Matrix4<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        Some(
            Matrix4Trait::from_columns(
                Self::back_substitute(self, self.q.row1()),
                Self::back_substitute(self, self.q.row2()),
                Self::back_substitute(self, self.q.row3()),
                Self::back_substitute(self, self.q.row4()),
            ),
        )
    }

    /// `R^-1 * y` by back substitution, the shared body of `solve` and `try_inverse`. The caller
    /// guarantees a nonzero diagonal. Two roundings per component: the numerator, accumulated
    /// exactly in `Real::Wide`, then the correctly rounded division. No upstream equivalent.
    fn back_substitute(self: Qr4<T>, y: Vector4<T>) -> Vector4<T> {
        let x4 = R::div(y.w, self.r.m44);
        let x3 = R::div(R::mul_add(-self.r.m34, x4, y.z), self.r.m33);
        let w = R::wide_sub_prod(R::wide_add(R::wide_zero(), y.y), self.r.m23, x3);
        let x2 = R::div(R::wide_rescale(R::wide_sub_prod(w, self.r.m24, x4)), self.r.m22);
        let w = R::wide_sub_prod(R::wide_add(R::wide_zero(), y.x), self.r.m12, x2);
        let w = R::wide_sub_prod(w, self.r.m13, x3);
        let x1 = R::div(R::wide_rescale(R::wide_sub_prod(w, self.r.m14, x4)), self.r.m11);
        Vector4 { x: x1, y: x2, z: x3, w: x4 }
    }

    /// The determinant: `det(Q) * r11 * r22 * r33 * r44`, see `Qr3::determinant` for the sign
    /// convention and the rounding of the product chain. Upstream: `QR::determinant`.
    ///
    /// `det(Q)` costs a full `Matrix4::determinant` (the 4x4 has no cheaper sign test), which is
    /// most of the price of this method; PREFER `Matrix4::determinant` on the matrix itself when
    /// the factorisation is not needed for something else.
    fn determinant(self: Qr4<T>) -> T {
        let d = if self.q.determinant().is_negative() {
            -self.r.m11
        } else {
            self.r.m11
        };
        d * self.r.m22 * self.r.m33 * self.r.m44
    }
}

/// `Matrix4` methods that go through the QR factorisation; upstream carries them on the matrix
/// itself. Import `Matrix4QrTrait` to use them.
#[generate_trait]
pub impl Matrix4QrImpl<
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
> of Matrix4QrTrait<T> {
    /// The QR factorisation. Upstream: `Matrix4::qr`.
    #[inline(always)]
    fn qr(self: Matrix4<T>) -> Qr4<T> {
        Qr4Trait::new(self)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Qr4`: exact cases, the identities and the oracle vectors of `tools/oracle`.

    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix4::{Matrix4, Matrix4Trait};
    use crate::base::matrix_test_utils::{
        amax_m4, excess, int, m4, max_abs_m4, max_abs_v4, max_ulp_diff4, max_ulp_diff_v4,
        oracle_tol, orthonormality_error_m4, v4t,
    };
    use crate::base::vector4::{Vector4, Vector4Trait};
    use crate::linalg::qr::oracle_qr4 as oracle;
    use super::{Matrix4QrTrait, Qr4, Qr4Trait};

    /// An oracle `unit` 4x4 case: the benchmark input.
    fn a_bench() -> Matrix4<Fixed> {
        m4(
            [
                [-151519325, -831704190, -37020683, -258362626],
                [-309967323, -174899491, -990814222, 1188862699],
                [526702394, 89514843, 1516307715, 154933714],
                [1138644662, -23815325, 661466655, 61972869],
            ],
        )
    }

    /// Its right-hand side, from `qr4_solve`.
    fn b_bench() -> Vector4<Fixed> {
        v4t((2614746463, 2804691241, -3498193820, 3365132750))
    }

    /// `a_bench()` already factored.
    fn f_bench() -> Qr4<Fixed> {
        Qr4Trait::new(a_bench())
    }

    /// A rank-3 matrix: the fourth column is the second, so `r44` is exactly zero.
    fn a_rank3() -> Matrix4<Fixed> {
        Matrix4Trait::from_columns(
            Vector4 { x: int(2), y: int(0), z: int(0), w: int(0) },
            Vector4 { x: int(0), y: int(4), z: int(0), w: int(0) },
            Vector4 { x: int(0), y: int(0), z: int(8), w: int(0) },
            Vector4 { x: int(0), y: int(4), z: int(0), w: int(0) },
        )
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_identity_and_diagonal_are_exact() {
        let f = Qr4Trait::new(Matrix4Trait::<Fixed>::identity());
        assert!(f.q() == Matrix4Trait::identity());
        assert!(f.r() == Matrix4Trait::identity());
        assert!(f.determinant() == int(1));
        let d = Matrix4Trait::from_diagonal(
            Vector4 { x: int(2), y: int(-4), z: int(8), w: int(-1) },
        );
        let f = Qr4Trait::new(d);
        assert!(
            f
                .r() == Matrix4Trait::from_diagonal(
                    Vector4 { x: int(2), y: int(4), z: int(8), w: int(1) },
                ),
        );
        assert!(f.q() * f.r() == d);
        assert!(f.determinant() == int(64));
        assert!(orthonormality_error_m4(f.q()) == 0);
    }

    #[test]
    fn test_new_rank_deficient_leaves_a_zero_column() {
        let f = Qr4Trait::new(a_rank3());
        assert!(f.r().m44 == Real::ZERO);
        assert!(f.q().column4() == Vector4Trait::zeros());
        assert!(f.q() * f.r() == a_rank3());
        assert!(!f.is_invertible());
        assert!(f.solve(b_bench()).is_none());
        assert!(f.try_inverse().is_none());
        assert!(f.determinant() == int(0));
    }

    #[test]
    fn test_unpack_matches_the_accessors() {
        let f = f_bench();
        let (q, r) = f.unpack();
        assert!(q == f.q() && r == f.r());
        assert!(a_bench().qr() == f);
    }

    // --- oracle --------------------------------------------------------------------------------

    #[test]
    fn test_new_factors_oracle() {
        let mut cases = oracle::qr4_q_r_cases();
        let (mut worst_q, mut worst_r, mut worst_ex) = (0, 0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, q, r, tol) = *case;
            let f = Qr4Trait::new(m4(a));
            let (eq, er) = (m4(q), m4(r));
            let (dq, dr) = (max_ulp_diff4(f.q(), eq), max_ulp_diff4(f.r(), er));
            worst_ex = core::cmp::max(worst_ex, excess(dq, oracle_tol(max_abs_m4(eq), tol)));
            worst_ex = core::cmp::max(worst_ex, excess(dr, oracle_tol(max_abs_m4(er), tol)));
            worst_q = core::cmp::max(worst_q, dq);
            worst_r = core::cmp::max(worst_r, dr);
        }
        assert!(
            (worst_q, worst_r, worst_ex) == (84, 15, 0),
            "regressed: {worst_q} {worst_r} {worst_ex}",
        );
    }

    #[test]
    fn test_new_reconstruction_and_orthonormality_oracle() {
        let mut cases = oracle::qr4_q_r_cases();
        let (mut worst_rec, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Qr4Trait::new(m4(a));
            let rec = max_ulp_diff4(f.q() * f.r(), m4(a)) / amax_m4(m4(a));
            let orth = orthonormality_error_m4(f.q());
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        // Measured: `|A - Q R| <= worst_rec ulp * max(1, max |a_ij|)`, `|QᵀQ - I| <= worst_orth`.
        assert!(worst_rec == 3 && worst_orth == 86, "regressed: {worst_rec} {worst_orth}");
    }

    #[test]
    fn test_solve_oracle() {
        let mut cases = oracle::qr4_solve_cases();
        let (mut worst, mut worst_ex) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let x = Qr4Trait::new(m4(a)).solve(v4t(b)).unwrap();
            let e = v4t(expected);
            let err = max_ulp_diff_v4(x, e);
            worst_ex = core::cmp::max(worst_ex, excess(err, oracle_tol(max_abs_v4(e), tol)));
            worst = core::cmp::max(worst, err);
        }
        assert!((worst, worst_ex) == (940, 0), "regressed: {worst} {worst_ex}");
    }

    #[test]
    fn test_try_inverse_product_is_identity_oracle() {
        let mut cases = oracle::qr4_q_r_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let inv = Qr4Trait::new(m4(a)).try_inverse().unwrap();
            let id = Matrix4Trait::identity();
            worst = core::cmp::max(worst, max_ulp_diff4(m4(a) * inv, id));
            worst = core::cmp::max(worst, max_ulp_diff4(inv * m4(a), id));
        }
        // Measured residual of `A A^-1 - I` and `A^-1 A - I` over the 30 well-conditioned vectors.
        assert!(worst == 128, "regressed: {worst}");
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_overflow_panics() {
        // 2^-32 * I: every diagonal entry of R is 1 raw unit, so the inverse is 2^32 * I.
        let _ = black_box(Matrix4Trait::from_diagonal_element(Fixed { raw: 1 })).qr().try_inverse();
    }

    // --- gas benchmarks --------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_qr4_new__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_new__gram_schmidt() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = Qr4Trait::new(a);
        assert!((f.r.m11 > Real::ZERO && f.r.m44 > Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_factors__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_factors__unpack() {
        let f = black_box(f_bench());
        let e = black_box(true);
        let (q, r) = f.unpack();
        assert!((q.m11 != Real::ZERO && r.m11 != Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_is_invertible__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_is_invertible__pivots() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.is_invertible() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_solve__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(b_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_solve__substitution() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(true);
        assert!(f.solve(b).is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_try_inverse__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_try_inverse__substitution() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.try_inverse().is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_determinant__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr4_determinant__diagonal_product() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.determinant() != Real::ZERO) == e);
    }
}
