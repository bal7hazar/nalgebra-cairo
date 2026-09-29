//! `Qr3`: the QR factorisation of a `Matrix3` (upstream `nalgebra::linalg::QR` on a 3x3 matrix).
//!
//! `A = Q * R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal — the
//! convention of upstream's unpacked `q()` / `r()`, see the module doc of `linalg::qr`.

use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::matrix_tr_mul::MatrixTrMul;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::internal::base::solve::SolveKernel;
use nalgebra_static3::base::matrix3::Matrix3Trait;
use nalgebra_static3::internal::base::matrix3::Matrix3InternalTrait;
use simba::scalar::Real;
use crate::internal::linalg::qr::qr3::Qr3InternalTrait;

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
        let (q11, q21, q31) = if r11 == R::zero() {
            (R::zero(), R::zero(), R::zero())
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
        let (q12, q22, q32) = if r22 == R::zero() {
            (R::zero(), R::zero(), R::zero())
        } else {
            R::div3(b21, b22, b23, r22)
        };
        let r23 = R::sum_prod3(q12, b31, q22, b32, q32, b33);
        let c31 = R::mul_add(-r23, q12, b31);
        let c32 = R::mul_add(-r23, q22, b32);
        let c33 = R::mul_add(-r23, q32, b33);
        let r33 = R::norm3(c31, c32, c33);
        let (q13, q23, q33) = if r33 == R::zero() {
            (R::zero(), R::zero(), R::zero())
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
                m21: R::zero(),
                m31: R::zero(),
                m12: r12,
                m22: r22,
                m32: R::zero(),
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

    /// The upper triangular factor `R`, consuming the factorisation: `r()`. Exact. Upstream:
    /// `QR::unpack_r`.
    #[inline(always)]
    fn unpack_r(self: Qr3<T>) -> Matrix3<T> {
        self.r
    }

    /// The factors as the factorisation stores them, `(Q, R)`: modified Gram-Schmidt keeps both
    /// in full, where upstream's Householder storage packs the reflectors and `R` into one matrix
    /// (the representation is internal on both sides). Exact. Upstream: `QR::qr_internal`
    /// (`#[doc(hidden)]`).
    #[inline(always)]
    fn qr_internal(self: Qr3<T>) -> (Matrix3<T>, Matrix3<T>) {
        (self.q, self.r)
    }

    /// `rhs = Qᵀ * rhs` in place, for any `rhs` with 3 rows (a vector or a matrix): ONE fused sum
    /// of products per entry (`MatrixTrMul::tr_mul`, floored once). Upstream: `QR::q_tr_mul`,
    /// which applies the Householder reflections one after the other (a rounding each).
    fn q_tr_mul<B, impl K: SolveKernel<Matrix3<T>, B>, +Drop<B>>(self: Qr3<T>, ref rhs: B) {
        rhs = K::tr_mul_rhs(self.q, rhs);
    }

    /// Overwrites `b` (any shape with 3 rows: a vector or a matrix) with the solution of `A * x =
    /// b` and returns `true`, or returns `false` and leaves `b` unchanged when a diagonal entry of
    /// `R` is exactly zero (`is_invertible`; upstream returns `false` after overwriting `b` with
    /// `Qᵀ b`). `x = R⁻¹ (Qᵀ b)`: one fused `tr_mul`, then back substitution (one floor and
    /// one correctly rounded division per component, one prepared divisor per row from 3 columns
    /// on); on a vector it is bit-identical to `solve`. Panics on overflow. Upstream:
    /// `QR::solve_mut`.
    fn solve_mut<B, impl K: SolveKernel<Matrix3<T>, B>, +Drop<B>>(self: Qr3<T>, ref b: B) -> bool {
        if !Self::is_invertible(self) {
            return false;
        }
        b = K::upper(self.r, K::tr_mul_rhs(self.q, b));
        true
    }

    /// Whether the factorisation is invertible: all 3 diagonal entries of `R` are EXACTLY nonzero
    /// (no epsilon), like upstream's `QR::is_invertible`. Equivalently, `Q` is orthonormal.
    #[inline(always)]
    fn is_invertible(self: Qr3<T>) -> bool {
        self.r.m11 != R::zero() && self.r.m22 != R::zero() && self.r.m33 != R::zero()
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a diagonal entry of `R`
    /// is exactly zero (`is_invertible`). Upstream: `QR::solve`.
    ///
    /// `x = R^-1 (Qᵀ b)`: one fused `tr_mul` for `Qᵀ b` (one rounding per component), then
    /// back substitution. Each component of `x` costs TWO roundings — the numerator, accumulated
    /// exactly in `Real::Wide` whatever the number of terms, then the correctly rounded division by
    /// the diagonal entry. Panics with the scalar's overflow error if a component of `x` does not
    /// fit.
    fn solve(self: Qr3<T>, b: Vector3<T>) -> Option<Vector3<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let y = self.q.tr_mul(b);
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
