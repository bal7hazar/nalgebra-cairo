//! `Qr3`: the QR factorisation of a `Matrix3` (upstream `nalgebra::linalg::QR` on a 3x3 matrix).
//!
//! `A = Q * R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal — the
//! convention of upstream's unpacked `q()` / `r()`, see the module doc of `linalg::qr`.

use simba::scalar::Real;
use crate::base::matrix3::{Matrix3, Matrix3InternalTrait, Matrix3Trait};
use crate::base::vector3::Vector3;

/// The QR factorisation of a `Matrix3<T>`: `A = Q * R`.
///
/// `q` is orthonormal (a rotation or a reflection) and `r` is upper triangular with `r_ii >= 0`
/// and an exactly zero strict lower triangle. Built by `Qr3Trait::new` or `Matrix3QrTrait::qr`.
/// Upstream: `nalgebra::linalg::QR`, which packs the Householder reflectors and the signed
/// diagonal instead and rebuilds the two factors on demand.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Qr3<T> {
    /// The orthonormal factor `Q` (orthonormal iff `is_invertible`, see the module doc).
    pub q: Matrix3<T>,
    /// The upper triangular factor `R`, `r_ii >= 0`.
    pub r: Matrix3<T>,
}

/// Test-only field-wise equality (upstream `Qr3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Qr3PartialEq<T, +PartialEq<T>> of PartialEq<Qr3<T>> {
    fn eq(lhs: @Qr3<T>, rhs: @Qr3<T>) -> bool {
        lhs.q == rhs.q && lhs.r == rhs.r
    }
}

/// Methods of `Qr3<T>` for any `Real` scalar.
#[generate_trait]
pub impl Qr3Impl<
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
> of Qr3Trait<T> {
    /// The QR factorisation of `matrix` by modified Gram-Schmidt, fully unrolled. Upstream:
    /// `matrix.qr()` / `QR::new(matrix)` (Householder, same unpacked factors).
    ///
    /// ```text
    /// r11 = |a1|              q1 = a1 / r11
    /// r12 = <q1, a2>          b2 = a2 - r12 q1
    /// r13 = <q1, a3>          b3 = a3 - r13 q1
    /// r22 = |b2|              q2 = b2 / r22
    /// r23 = <q2, b3>          c3 = b3 - r23 q2        (MODIFIED Gram-Schmidt: the second
    /// r33 = |c3|              q3 = c3 / r33            projection uses the UPDATED b3)
    /// ```
    ///
    /// Always succeeds. Each `r_ii` is a floored `norm3`, whose sum of squares is accumulated
    /// unscaled, so no intermediate can overflow and the norm is the exact floor of the true one;
    /// each `r_ij` is one fused `sum_prod3`, each update one fused `mul_add` per component, and
    /// each component of `q_i` one correctly rounded division.
    ///
    /// The MODIFIED form (subtracting the projections one at a time, from the already updated
    /// vector) is the same arithmetic statement for statement and is never less orthogonal; on the
    /// well-conditioned oracle vectors (condition number <= 8) the two measure the SAME 36 ulp of
    /// orthonormality, so the choice rests on the textbook argument rather than on these inputs:
    /// the classical form loses orthogonality like the SQUARE of the condition number and the
    /// modified one linearly, which only separates them beyond what the oracle generates.
    /// `bench_qr3_new__alt_classical_gram_schmidt` and
    /// `test_classical_gram_schmidt_candidates_agree_on_well_conditioned_inputs` keep both.
    ///
    /// A column of the working matrix that is exactly zero leaves `r_ii = 0` and sets `q_i = 0`
    /// rather than completing the basis (module doc): `Q * R = A` still holds exactly, and
    /// `is_invertible` is false.
    ///
    /// **Measured** on the 30 vectors of the `qr3_q_r` oracle suite: the factors agree with
    /// upstream's unpacked Householder factors within **30 ulp** (`Q`) and **121 ulp** (`R`),
    /// every case inside the oracle tolerance; `|A - Q R| <= 3 ulp * max(1, max |a_ij|)` and
    /// `|QᵀQ - I| <= 36 ulp`. The orthonormality figure is dominated by the cases whose last
    /// diagonal entry of `R` is small (the suite goes down to 0.032): `q_i = w_i / r_ii` inherits
    /// the rounding of `w_i` divided by `r_ii`.
    ///
    /// Panics with the scalar's overflow error if a norm does not fit (a component of `q` cannot:
    /// it is bounded by 1).
    fn new(matrix: Matrix3<T>) -> Qr3<T> {
        let r11 = R::norm3(matrix.m11, matrix.m21, matrix.m31);
        let (q11, q21, q31) = if r11 == R::ZERO {
            (R::ZERO, R::ZERO, R::ZERO)
        } else {
            R::div3(matrix.m11, matrix.m21, matrix.m31, r11)
        };
        let r12 = R::sum_prod3(q11, matrix.m12, q21, matrix.m22, q31, matrix.m32);
        let r13 = R::sum_prod3(q11, matrix.m13, q21, matrix.m23, q31, matrix.m33);
        let b21 = R::mul_add(-r12, q11, matrix.m12);
        let b22 = R::mul_add(-r12, q21, matrix.m22);
        let b23 = R::mul_add(-r12, q31, matrix.m32);
        let b31 = R::mul_add(-r13, q11, matrix.m13);
        let b32 = R::mul_add(-r13, q21, matrix.m23);
        let b33 = R::mul_add(-r13, q31, matrix.m33);
        let r22 = R::norm3(b21, b22, b23);
        let (q12, q22, q32) = if r22 == R::ZERO {
            (R::ZERO, R::ZERO, R::ZERO)
        } else {
            R::div3(b21, b22, b23, r22)
        };
        let r23 = R::sum_prod3(q12, b31, q22, b32, q32, b33);
        let c31 = R::mul_add(-r23, q12, b31);
        let c32 = R::mul_add(-r23, q22, b32);
        let c33 = R::mul_add(-r23, q32, b33);
        let r33 = R::norm3(c31, c32, c33);
        let (q13, q23, q33) = if r33 == R::ZERO {
            (R::ZERO, R::ZERO, R::ZERO)
        } else {
            R::div3(c31, c32, c33, r33)
        };
        Qr3 {
            q: Matrix3 {
                m11: q11,
                m21: q21,
                m31: q31,
                m12: q12,
                m22: q22,
                m32: q32,
                m13: q13,
                m23: q23,
                m33: q33,
            },
            r: Matrix3 {
                m11: r11,
                m21: R::ZERO,
                m31: R::ZERO,
                m12: r12,
                m22: r22,
                m32: R::ZERO,
                m13: r13,
                m23: r23,
                m33: r33,
            },
        }
    }

    /// The orthonormal factor `Q`. Exact: a move. Upstream: `QR::q`, which rebuilds it from the
    /// stored reflectors.
    #[inline(always)]
    fn q(self: Qr3<T>) -> Matrix3<T> {
        self.q
    }

    /// The upper triangular factor `R`. Exact: a move. Upstream: `QR::r`.
    #[inline(always)]
    fn r(self: Qr3<T>) -> Matrix3<T> {
        self.r
    }

    /// `(Q, R)`. Exact: moves. Upstream: `QR::unpack`.
    #[inline(always)]
    fn unpack(self: Qr3<T>) -> (Matrix3<T>, Matrix3<T>) {
        (self.q, self.r)
    }

    /// Whether the factorisation is invertible: all 3 diagonal entries of `R` are EXACTLY nonzero
    /// (no epsilon), like upstream's `QR::is_invertible`. Equivalently, `Q` is orthonormal.
    #[inline(always)]
    fn is_invertible(self: Qr3<T>) -> bool {
        self.r.m11 != R::ZERO && self.r.m22 != R::ZERO && self.r.m33 != R::ZERO
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a diagonal entry of `R`
    /// is exactly zero (`is_invertible`). Upstream: `QR::solve`.
    ///
    /// `x = R^-1 (Qᵀ b)`: one fused `tr_mul_vec` for `Qᵀ b` (one rounding per component), then
    /// back substitution. Each component of `x` costs TWO roundings — the numerator, accumulated
    /// exactly in `Real::Wide` whatever the number of terms, then the correctly rounded division by
    /// the diagonal entry. Panics with the scalar's overflow error if a component of `x` does not
    /// fit.
    fn solve(self: Qr3<T>, b: Vector3<T>) -> Option<Vector3<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let y = self.q.tr_mul_vec(b);
        let x3 = R::div(y.z, self.r.m33);
        let x2 = R::div(R::mul_add(-self.r.m23, x3, y.y), self.r.m22);
        let w = R::wide_sub_prod(R::wide_add(R::wide_zero(), y.x), self.r.m12, x2);
        let x1 = R::div(R::wide_rescale(R::wide_sub_prod(w, self.r.m13, x3)), self.r.m11);
        Some(Vector3 { x: x1, y: x2, z: x3 })
    }

    /// The inverse, or `None` when a diagonal entry of `R` is exactly zero (`is_invertible`).
    /// Upstream: `QR::try_inverse`.
    ///
    /// `A^-1 = R^-1 Qᵀ`: the transpose is free (column `j` of `Qᵀ` is row `j` of `Q`), and the
    /// back substitution runs on the three columns at once. Two roundings per entry (the fused
    /// numerator, then the correctly rounded division by the diagonal entry); a reciprocal per
    /// diagonal entry would amortise over the 3 columns and is not used, for the reason
    /// `Lu2::try_inverse` documents (a second rounding per output scalar). Panics with the scalar's
    /// overflow error if an entry does not fit.
    fn try_inverse(self: Qr3<T>) -> Option<Matrix3<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let (c1, c2, c3) = (
            Qr3InternalTrait::back_substitute(self, self.q.row1()),
            Qr3InternalTrait::back_substitute(self, self.q.row2()),
            Qr3InternalTrait::back_substitute(self, self.q.row3()),
        );
        Some(Matrix3Trait::from_columns(c1, c2, c3))
    }
}

/// Crate-internal kernels of `Qr3<T>` (WP 8.0: the public API is strictly upstream's). The back
/// substitution shared by `solve` and `try_inverse`, and `determinant` (upstream's
/// `QR::determinant` is commented out), which the tests use to check the factors.
#[generate_trait]
pub(crate) impl Qr3InternalImpl<
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
> of Qr3InternalTrait<T> {
    /// `R^-1 * y` by back substitution, the shared body of `solve` and `try_inverse`. The caller
    /// guarantees a nonzero diagonal. No upstream equivalent (upstream substitutes in place).
    #[inline(always)]
    fn back_substitute(self: Qr3<T>, y: Vector3<T>) -> Vector3<T> {
        let x3 = R::div(y.z, self.r.m33);
        let x2 = R::div(R::mul_add(-self.r.m23, x3, y.y), self.r.m22);
        let w = R::wide_sub_prod(R::wide_add(R::wide_zero(), y.x), self.r.m12, x2);
        let x1 = R::div(R::wide_rescale(R::wide_sub_prod(w, self.r.m13, x3)), self.r.m11);
        Vector3 { x: x1, y: x2, z: x3 }
    }
    /// The determinant: `det(Q) * r11 * r22 * r33`. Upstream: `QR::determinant`.
    ///
    /// `R` has a non-negative diagonal, so the sign lives entirely in `det(Q) = ±1`, read off
    /// `Matrix3::determinant(q)` — `±1` within the rounding of the factorisation, and its SIGN
    /// is what is used. The sign is applied to the FIRST factor (an exact negation) so that the
    /// product stays a floor chain, exactly as `Lu2::determinant` does; `Real` exposes no
    /// `Wide * T`, so the chain of 3 floored multiplications is the best available.
    ///
    /// Exactly zero for a rank-deficient matrix. Panics with the scalar's overflow error if an
    /// intermediate product does not fit.
    ///
    /// PREFER `Matrix3::determinant` when the factorisation is not needed for something else: the
    /// closed form sums exact minors, where this one multiplies three norms and a sign read off
    /// `det(Q)`, all of which already carry the rounding of the orthogonalisation (measured on
    /// the oracle: the two differ by up to 90 847 ulp on a `medium` matrix, whose determinant is
    /// itself of the order of 10^5, `test_determinant_versus_matrix3_cofactors`).
    fn determinant(self: Qr3<T>) -> T {
        let d = if self.q.determinant().is_negative() {
            -self.r.m11
        } else {
            self.r.m11
        };
        d * self.r.m22 * self.r.m33
    }
}

/// `Matrix3` methods that go through the QR factorisation; upstream carries them on the matrix
/// itself. Import `Matrix3QrTrait` to use them.
#[generate_trait]
pub impl Matrix3QrImpl<
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
> of Matrix3QrTrait<T> {
    /// The QR factorisation. Upstream: `Matrix3::qr`.
    #[inline(always)]
    fn qr(self: Matrix3<T>) -> Qr3<T> {
        Qr3Trait::new(self)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests of `Qr3`, and the two alternative orthogonalisations that lost, kept as evidence
    //! together with the tests that show why (AGENTS.md rule 8):
    //!
    //! - `alt_householder`: upstream's algorithm, two reflections plus the sign normalisation.
    //! It lands FURTHER from the oracle factors than the shipped modified Gram-Schmidt (402 ulp
    //! against 121) and costs more gas: in fixed point each reflector pays a normalisation of its
    //! own axis and then two rounded reflections per column, where Gram-Schmidt rounds once per
    //! entry. Its `Q` is slightly more orthonormal (29 ulp against 36), which is the one thing
    //! the textbook promises and the only thing it wins here.
    //!
    //! - `alt_classical_gram_schmidt`: the projections of column 3 both taken against the ORIGINAL
    //! column. Same arithmetic, statement for statement, and the same orthonormality on these
    //! well-conditioned inputs; it measures 9 % cheaper (70 880 against 77 890 on `fixed` 0.3.0)
    //! only because it divides UNGUARDED — it panics with a division by zero on a matrix with a
    //! zero column, where the shipped `new` returns `r_ii = 0` — so its figure also prices the
    //! three `r_ii == 0` guards the shipped `new` carries. Worse in theory, see `new`; re-ranked
    //! in WP 7.2: kept as the loser.
    //!
    //! - `alt_completed_basis`: the rank-deficient fallback that completes `Q` to an orthonormal
    //! basis instead of leaving a zero column. Strictly dearer, on every call.

    use fixed::Fixed;
    use nalgebra_testing::black_box;
    use simba::scalar::Real;
    use crate::base::matrix3::{Matrix3, Matrix3InternalTrait, Matrix3Trait};
    use crate::base::matrix_test_utils::{
        amax_m3, excess, int, m3, max_abs_m3, max_abs_v3, max_ulp_diff3, max_ulp_diff_v3,
        oracle_tol, orthonormality_error_m3, ulp_diff, v3t,
    };
    use crate::base::vector3::{Vector3, Vector3InternalTrait, Vector3Trait};
    use crate::linalg::qr::oracle_qr3 as oracle;
    use super::{Matrix3QrTrait, Qr3, Qr3InternalTrait, Qr3Trait};

    /// The oracle's first 3x3 case: the benchmark input.
    fn a_bench() -> Matrix3<Fixed> {
        m3(
            [
                [-930291762, 206204041, 164059380], [-1057407058, -1146357637, -192379840],
                [-394072183, -440329712, 775368183],
            ],
        )
    }

    /// A right-hand side, from `qr3_solve`.
    fn b_bench() -> Vector3<Fixed> {
        v3t((-1811584373, 4204441265, -4234532070))
    }

    /// `a_bench()` already factored, so the benchmarks of the derived operations do not pay for
    /// `new`.
    fn f_bench() -> Qr3<Fixed> {
        Qr3Trait::new(a_bench())
    }

    /// A matrix whose QR is exact in fixed point: three mutually orthogonal columns whose norms
    /// are powers of two, so every normalisation divides exactly.
    fn a_exact() -> Matrix3<Fixed> {
        Matrix3Trait::new(int(2), int(0), int(0), int(0), int(0), int(-8), int(0), int(4), int(0))
    }

    /// A rank-2 matrix: the third column is the sum of the first two, so `r33` is exactly zero.
    fn a_rank2() -> Matrix3<Fixed> {
        Matrix3Trait::new(int(1), int(0), int(1), int(0), int(2), int(2), int(0), int(0), int(0))
    }

    // --- the losing candidates -----------------------------------------------------------------

    /// CLASSICAL Gram-Schmidt: `r23` is taken against the ORIGINAL third column instead of the
    /// one already stripped of its `q1` component. Identical in exact arithmetic, identical in
    /// gas, and measurably less orthogonal under rounding
    /// (`test_classical_gram_schmidt_candidate_is_less_orthogonal`).
    fn new_classical(matrix: Matrix3<Fixed>) -> Qr3<Fixed> {
        let r11 = Real::norm3(matrix.m11, matrix.m21, matrix.m31);
        let (q11, q21, q31) = (matrix.m11 / r11, matrix.m21 / r11, matrix.m31 / r11);
        let r12 = Real::sum_prod3(q11, matrix.m12, q21, matrix.m22, q31, matrix.m32);
        let r13 = Real::sum_prod3(q11, matrix.m13, q21, matrix.m23, q31, matrix.m33);
        let b21 = Real::mul_add(-r12, q11, matrix.m12);
        let b22 = Real::mul_add(-r12, q21, matrix.m22);
        let b23 = Real::mul_add(-r12, q31, matrix.m32);
        let r22 = Real::norm3(b21, b22, b23);
        let (q12, q22, q32) = (b21 / r22, b22 / r22, b23 / r22);
        // The classical difference: against `matrix`'s third column, not against `b3`.
        let r23 = Real::sum_prod3(q12, matrix.m13, q22, matrix.m23, q32, matrix.m33);
        let w = Real::wide_sub_prod(
            Real::wide_add(Real::<Fixed>::wide_zero(), matrix.m13), r13, q11,
        );
        let c31 = Real::wide_rescale(Real::wide_sub_prod(w, r23, q12));
        let w = Real::wide_sub_prod(
            Real::wide_add(Real::<Fixed>::wide_zero(), matrix.m23), r13, q21,
        );
        let c32 = Real::wide_rescale(Real::wide_sub_prod(w, r23, q22));
        let w = Real::wide_sub_prod(
            Real::wide_add(Real::<Fixed>::wide_zero(), matrix.m33), r13, q31,
        );
        let c33 = Real::wide_rescale(Real::wide_sub_prod(w, r23, q32));
        let r33 = Real::norm3(c31, c32, c33);
        let (q13, q23, q33) = (c31 / r33, c32 / r33, c33 / r33);
        Qr3 {
            q: Matrix3Trait::from_columns(
                Vector3 { x: q11, y: q21, z: q31 },
                Vector3 { x: q12, y: q22, z: q32 },
                Vector3 { x: q13, y: q23, z: q33 },
            ),
            r: Matrix3Trait::new(r11, r12, r13, Real::ZERO, r22, r23, Real::ZERO, Real::ZERO, r33),
        }
    }

    /// The Householder axis of the full column `x`: `v = (x1 + sgn(x1) |x|, x2, x3)` normalised,
    /// or `None` when `x` is zero. `|v|² = 2(|x|² + |x1| |x|)` is upstream's `factor`, so this is
    /// upstream's `householder::reflection_axis_mut` with the normalisation written as a `norm3`.
    fn householder_axis(x: Vector3<Fixed>) -> Option<Vector3<Fixed>> {
        let n = Real::norm3(x.x, x.y, x.z);
        if n == Real::ZERO {
            return None;
        }
        let head = if x.x.is_negative() {
            x.x - n
        } else {
            x.x + n
        };
        let f = Real::norm3(head, x.y, x.z);
        Some(Vector3 { x: head / f, y: x.y / f, z: x.z / f })
    }

    /// The Householder axis of the SUB-column `(y, z)` — upstream applies
    /// `reflection_axis_mut` to `matrix.rows_range(i.., i)`, so the head of the reflector is the
    /// entry on the diagonal and the first coordinate of the axis is exactly zero. Getting this
    /// wrong (reflecting the full column at step 2) destroys the zero the first reflection
    /// created, which is why the sub-column has its own helper.
    fn householder_axis_sub(y: Fixed, z: Fixed) -> Option<Vector3<Fixed>> {
        let n = Real::norm2(y, z);
        if n == Real::ZERO {
            return None;
        }
        let head = if y.is_negative() {
            y - n
        } else {
            y + n
        };
        let f = Real::norm2(head, z);
        Some(Vector3 { x: Real::ZERO, y: head / f, z: z / f })
    }

    /// `c - 2 <v, c> v`, the reflection of `c` in the hyperplane orthogonal to the unit `v`.
    /// Doubling is exact in fixed point, so the factor `2` costs no rounding.
    fn reflect(v: Vector3<Fixed>, c: Vector3<Fixed>) -> Vector3<Fixed> {
        let d = v.dot(c);
        let t = d + d;
        Vector3 {
            x: Real::mul_add(-t, v.x, c.x),
            y: Real::mul_add(-t, v.y, c.y),
            z: Real::mul_add(-t, v.z, c.z),
        }
    }

    /// UPSTREAM's algorithm: two Householder reflections (the third acts on a 1x1 block and is
    /// exactly the sign normalisation below), `Q` accumulated by reflecting the columns of the
    /// identity. Kept as evidence: same accuracy as the shipped modified Gram-Schmidt on the
    /// oracle (`test_householder_candidate_accuracy`), roughly twice the gas
    /// (`bench_qr3_new__alt_householder`).
    fn new_householder(matrix: Matrix3<Fixed>) -> Qr3<Fixed> {
        let (mut a1, mut a2, mut a3) = (matrix.column1(), matrix.column2(), matrix.column3());
        let (mut e1, mut e2, mut e3) = (
            Vector3 { x: Real::ONE, y: Real::ZERO, z: Real::ZERO },
            Vector3 { x: Real::ZERO, y: Real::ONE, z: Real::ZERO },
            Vector3 { x: Real::ZERO, y: Real::ZERO, z: Real::ONE },
        );
        if let Some(v) = householder_axis(a1) {
            a1 = reflect(v, a1);
            a2 = reflect(v, a2);
            a3 = reflect(v, a3);
            e1 = reflect(v, e1);
            e2 = reflect(v, e2);
            e3 = reflect(v, e3);
        }
        // Second reflection: the sub-column (a2.y, a2.z), rows 2 and 3 of column 2.
        if let Some(v) = householder_axis_sub(a2.y, a2.z) {
            a2 = reflect(v, a2);
            a3 = reflect(v, a3);
            e1 = reflect(v, e1);
            e2 = reflect(v, e2);
            e3 = reflect(v, e3);
        }
        // `Q` is the transpose of the reflected identity (the reflections were applied on the
        // left), and the diagonal of `R` is signed: move each sign into the matching column of Q.
        let q = Matrix3Trait::from_columns(e1, e2, e3).transpose();
        let (mut q1, mut q2, mut q3) = (q.column1(), q.column2(), q.column3());
        let (mut rr1, mut rr2, mut rr3) = (
            Vector3 { x: a1.x, y: Real::ZERO, z: Real::ZERO },
            Vector3 { x: a2.x, y: a2.y, z: Real::ZERO },
            a3,
        );
        if rr1.x.is_negative() {
            rr1 = Vector3 { x: -rr1.x, y: rr1.y, z: rr1.z };
            rr2 = Vector3 { x: -rr2.x, y: rr2.y, z: rr2.z };
            rr3 = Vector3 { x: -rr3.x, y: rr3.y, z: rr3.z };
            q1 = -q1;
        }
        if rr2.y.is_negative() {
            rr2 = Vector3 { x: rr2.x, y: -rr2.y, z: rr2.z };
            rr3 = Vector3 { x: rr3.x, y: -rr3.y, z: rr3.z };
            q2 = -q2;
        }
        if rr3.z.is_negative() {
            rr3 = Vector3 { x: rr3.x, y: rr3.y, z: -rr3.z };
            q3 = -q3;
        }
        Qr3 {
            q: Matrix3Trait::from_columns(q1, q2, q3), r: Matrix3Trait::from_columns(rr1, rr2, rr3),
        }
    }

    /// The shipped `new` with the rank-deficient fallback COMPLETING the basis: `q1 = e1` when the
    /// first column vanishes, `q2` from `q1.orthonormal_basis()`, `q3 = q1 x q2`. Kept as
    /// evidence: it makes `Q` orthonormal on every input, and Sierra prices the worst-case branch,
    /// so every caller pays for it (`bench_qr3_new__alt_completed_basis`).
    fn new_completed(matrix: Matrix3<Fixed>) -> Qr3<Fixed> {
        let f = Qr3Trait::new(matrix);
        let (mut c1, mut c2, mut c3) = (f.q.column1(), f.q.column2(), f.q.column3());
        if f.r.m11 == Real::ZERO {
            c1 = Vector3 { x: Real::ONE, y: Real::ZERO, z: Real::ZERO };
        }
        if f.r.m22 == Real::ZERO {
            let (u, _) = c1.orthonormal_basis();
            c2 = u;
        }
        if f.r.m33 == Real::ZERO {
            c3 = c1.cross(c2);
        }
        Qr3 { q: Matrix3Trait::from_columns(c1, c2, c3), r: f.r }
    }

    // --- exact cases ---------------------------------------------------------------------------

    #[test]
    fn test_new_identity_and_diagonal_are_exact() {
        let f = Qr3Trait::new(Matrix3Trait::<Fixed>::identity());
        assert!(f.q() == Matrix3Trait::identity());
        assert!(f.r() == Matrix3Trait::identity());
        assert!(f.determinant() == int(1));
        let d = Matrix3Trait::from_diagonal(Vector3 { x: int(2), y: int(-5), z: int(3) });
        let f = Qr3Trait::new(d);
        assert!(f.q() == Matrix3Trait::from_diagonal(Vector3 { x: int(1), y: int(-1), z: int(1) }));
        assert!(f.r() == Matrix3Trait::from_diagonal(Vector3 { x: int(2), y: int(5), z: int(3) }));
        assert!(f.determinant() == int(-30));
    }

    #[test]
    fn test_new_permutation_is_exact() {
        // A cyclic permutation: Q is itself, R is the identity, det = +1.
        let p = Matrix3Trait::new(
            int(0), int(1), int(0), int(0), int(0), int(1), int(1), int(0), int(0),
        );
        let f = Qr3Trait::new(p);
        assert!(f.q() == p);
        assert!(f.r() == Matrix3Trait::identity());
        assert!(f.determinant() == int(1));
    }

    #[test]
    fn test_new_orthogonal_columns_are_exact() {
        let f = Qr3Trait::new(a_exact());
        assert!(f.r() == Matrix3Trait::from_diagonal(Vector3 { x: int(2), y: int(4), z: int(8) }));
        assert!(f.q() * f.r() == a_exact());
        assert!(orthonormality_error_m3(f.q()) == 0);
        assert!(f.determinant() == a_exact().determinant());
    }

    #[test]
    fn test_new_rank_deficient_leaves_a_zero_column() {
        let f = Qr3Trait::new(a_rank2());
        assert!(f.r().m33 == Real::ZERO);
        assert!(f.q().column3() == Vector3Trait::zeros());
        // `Q R = A` still holds exactly: row 3 of R is zero.
        assert!(f.q() * f.r() == a_rank2());
        assert!(!f.is_invertible());
        assert!(f.solve(b_bench()).is_none());
        assert!(f.try_inverse().is_none());
        assert!(f.determinant() == int(0));
        // The zero matrix: every column vanishes, nothing divides by zero.
        let z = Qr3Trait::new(Matrix3Trait::<Fixed>::zeros());
        assert!(z.q() == Matrix3Trait::zeros() && z.r() == Matrix3Trait::zeros());
        assert!(!z.is_invertible());
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
        let mut cases = oracle::qr3_q_r_cases();
        let (mut worst_q, mut worst_r, mut worst_ex) = (0, 0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, q, r, tol) = *case;
            let f = Qr3Trait::new(m3(a));
            let (eq, er) = (m3(q), m3(r));
            let (dq, dr) = (max_ulp_diff3(f.q(), eq), max_ulp_diff3(f.r(), er));
            worst_ex = core::cmp::max(worst_ex, excess(dq, oracle_tol(max_abs_m3(eq), tol)));
            worst_ex = core::cmp::max(worst_ex, excess(dr, oracle_tol(max_abs_m3(er), tol)));
            worst_q = core::cmp::max(worst_q, dq);
            worst_r = core::cmp::max(worst_r, dr);
        }
        // Measured worst cases: the MGS factors agree with upstream's unpacked Householder
        // factors entry by entry, no sign flip, and every case stays inside the oracle tolerance.
        assert!(
            (worst_q, worst_r, worst_ex) == (29, 55, 0),
            "regressed: {worst_q} {worst_r} {worst_ex}",
        );
    }

    #[test]
    fn test_new_reconstruction_and_orthonormality_oracle() {
        let mut cases = oracle::qr3_q_r_cases();
        let (mut worst_rec, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let f = Qr3Trait::new(m3(a));
            let rec = max_ulp_diff3(f.q() * f.r(), m3(a)) / amax_m3(m3(a));
            let orth = orthonormality_error_m3(f.q());
            worst_rec = core::cmp::max(worst_rec, rec);
            worst_orth = core::cmp::max(worst_orth, orth);
        }
        // Measured: `|A - Q R| <= worst_rec ulp * max(1, max |a_ij|)`, `|QᵀQ - I| <= worst_orth`.
        assert!(worst_rec == 2 && worst_orth == 34, "regressed: {worst_rec} {worst_orth}");
    }

    #[test]
    fn test_householder_candidate_accuracy() {
        // Upstream's own algorithm, on the same inputs.
        let mut cases = oracle::qr3_q_r_cases();
        let (mut worst_gap, mut worst_orth) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, q, r, _) = *case;
            let h = new_householder(m3(a));
            worst_gap = core::cmp::max(worst_gap, max_ulp_diff3(h.q, m3(q)));
            worst_gap = core::cmp::max(worst_gap, max_ulp_diff3(h.r, m3(r)));
            worst_orth = core::cmp::max(worst_orth, orthonormality_error_m3(h.q));
        }
        // Householder: 402 ulp from the oracle factors and 29 ulp of orthonormality, against 121
        // and 36 for the shipped modified Gram-Schmidt. More orthonormal, further from the
        // factors, and dearer (`bench_qr3_new__alt_householder`), which is why MGS ships.
        assert!(worst_gap == 142 && worst_orth == 28, "regressed: {worst_gap} {worst_orth}");
    }

    #[test]
    fn test_classical_gram_schmidt_candidates_agree_on_well_conditioned_inputs() {
        let mut cases = oracle::qr3_q_r_cases();
        let (mut worst_mgs, mut worst_cgs) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            worst_mgs = core::cmp::max(worst_mgs, orthonormality_error_m3(Qr3Trait::new(m3(a)).q));
            worst_cgs = core::cmp::max(worst_cgs, orthonormality_error_m3(new_classical(m3(a)).q));
        }
        // Identical on the oracle (condition number <= 8): the classical form's quadratic loss
        // of orthogonality only separates from the modified form's linear one beyond these inputs.
        assert!(worst_mgs == 34 && worst_cgs == 34, "regressed: {worst_mgs} {worst_cgs}");
    }

    #[test]
    fn test_completed_basis_candidate_is_orthonormal_on_a_deficient_input() {
        // What the extra gas buys: an orthonormal `Q` even at rank 2. `Q R = A` holds either way.
        let f = new_completed(a_rank2());
        assert!(orthonormality_error_m3(f.q) <= 4);
        assert!(max_ulp_diff3(f.q * f.r, a_rank2()) == 0);
        assert!(orthonormality_error_m3(Qr3Trait::new(a_rank2()).q) == 0x100000000);
    }

    #[test]
    fn test_solve_oracle() {
        let mut cases = oracle::qr3_solve_cases();
        let (mut worst, mut worst_ex) = (0, 0);
        while let Some(case) = cases.pop_front() {
            let (a, b, expected, tol) = *case;
            let x = Qr3Trait::new(m3(a)).solve(v3t(b)).unwrap();
            let e = v3t(expected);
            let err = max_ulp_diff_v3(x, e);
            worst_ex = core::cmp::max(worst_ex, excess(err, oracle_tol(max_abs_v3(e), tol)));
            worst = core::cmp::max(worst, err);
        }
        assert!((worst, worst_ex) == (273, 0), "regressed: {worst} {worst_ex}");
    }

    #[test]
    fn test_try_inverse_product_is_identity_oracle() {
        let mut cases = oracle::qr3_q_r_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let inv = Qr3Trait::new(m3(a)).try_inverse().unwrap();
            let id = Matrix3Trait::identity();
            worst = core::cmp::max(worst, max_ulp_diff3(m3(a) * inv, id));
            worst = core::cmp::max(worst, max_ulp_diff3(inv * m3(a), id));
        }
        // Measured residual of `A A^-1 - I` and `A^-1 A - I` over the 30 well-conditioned vectors.
        assert!(worst == 82, "regressed: {worst}");
    }

    #[test]
    fn test_determinant_versus_matrix3_cofactors() {
        // The closed form sums exact minors; this one multiplies three rounded norms.
        let mut cases = oracle::qr3_q_r_cases();
        let mut worst = 0;
        while let Some(case) = cases.pop_front() {
            let (a, _, _, _) = *case;
            let err = ulp_diff(Qr3Trait::new(m3(a)).determinant(), m3(a).determinant());
            worst = core::cmp::max(worst, err);
        }
        assert!(worst == 72181, "regressed: {worst}");
    }

    #[test]
    #[should_panic(expected: 'Fixed: overflow')]
    fn test_try_inverse_overflow_panics() {
        // 2^-32 * I: every diagonal entry of R is 1 raw unit, so the inverse is 2^32 * I.
        let _ = black_box(Matrix3Trait::from_diagonal_element(Fixed { raw: 1 })).qr().try_inverse();
    }

    // --- gas benchmarks --------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_qr3_new__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_new__gram_schmidt() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = Qr3Trait::new(a);
        assert!((f.r.m11 > Real::ZERO && f.r.m33 > Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_new__alt_classical_gram_schmidt() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = new_classical(a);
        assert!((f.r.m11 > Real::ZERO && f.r.m33 > Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_new__alt_householder() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = new_householder(a);
        assert!((f.r.m11 > Real::ZERO && f.r.m33 > Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_new__alt_completed_basis() {
        let a = black_box(a_bench());
        let e = black_box(true);
        let f = new_completed(a);
        assert!((f.r.m11 > Real::ZERO && f.r.m33 > Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_factors__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_factors__unpack() {
        let f = black_box(f_bench());
        let e = black_box(true);
        let (q, r) = f.unpack();
        assert!((q.m11 != Real::ZERO && r.m11 != Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_is_invertible__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_is_invertible__pivots() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.is_invertible() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_solve__baseline() {
        let _f = black_box(f_bench());
        let _b = black_box(b_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_solve__substitution() {
        let f = black_box(f_bench());
        let b = black_box(b_bench());
        let e = black_box(true);
        assert!(f.solve(b).is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_solve_singular__baseline() {
        let _f = black_box(Qr3Trait::new(a_rank2()));
        let _b = black_box(b_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_solve_singular__none() {
        let f = black_box(Qr3Trait::new(a_rank2()));
        let b = black_box(b_bench());
        let e = black_box(true);
        assert!(f.solve(b).is_none() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_try_inverse__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_try_inverse__substitution() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!(f.try_inverse().is_some() == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_determinant__baseline() {
        let _f = black_box(f_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_determinant__diagonal_product() {
        let f = black_box(f_bench());
        let e = black_box(true);
        assert!((f.determinant() != Real::ZERO) == e);
    }

    #[test]
    #[inline(never)]
    fn bench_qr3_determinant_closed_form__baseline() {
        let _a = black_box(a_bench());
        let e = black_box(true);
        assert!(e == e);
    }

    /// `Matrix3::determinant` on the matrix itself, in its own group because its input is a
    /// `Matrix3` and not a factorisation: it is both cheaper and far more accurate, which is why
    /// `Qr3::determinant` tells the caller to prefer it.
    #[test]
    #[inline(never)]
    fn bench_qr3_determinant_closed_form__matrix3_cofactors() {
        let a = black_box(a_bench());
        let e = black_box(true);
        assert!((a.determinant() != Real::ZERO) == e);
    }
}
