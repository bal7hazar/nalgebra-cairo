//! `Qr4`: the QR factorisation of a `Matrix4` (upstream `nalgebra::linalg::QR` on a 4x4 matrix).
//!
//! `A = Q * R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal — the
//! convention of upstream's unpacked `q()` / `r()`, see the module doc of `linalg::qr`. The
//! algorithm is the modified Gram-Schmidt of `qr3`, one column longer; the study of the
//! alternatives (Householder, classical Gram-Schmidt, completed basis) lives there.

use nalgebra_core::base::matrix4::Matrix4;
use nalgebra_core::base::matrix_tr_mul::MatrixTrMul;
use nalgebra_core::base::vector4::Vector4;
use nalgebra_core::internal::base::solve::SolveKernel;
use nalgebra_static4::base::matrix4::Matrix4Trait;
use simba::scalar::Real;
use crate::internal::linalg::qr::qr4::Qr4InternalTrait;

/// The QR factorisation of a `Matrix4<T>`: `A = Q * R`.
///
/// `q` is orthonormal and `r` is upper triangular with `r_ii >= 0` and an exactly zero strict
/// lower triangle. Built by `Qr4Trait::new` or `Matrix4QrTrait::qr`.
/// Upstream: `nalgebra::linalg::QR`.
#[derive(Copy, Drop, Serde, Debug)]
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
        let q1 = Qr4InternalTrait::unit(a1, r11);
        let r12 = R::sum_prod4(q1.x, a2.x, q1.y, a2.y, q1.z, a2.z, q1.w, a2.w);
        let r13 = R::sum_prod4(q1.x, a3.x, q1.y, a3.y, q1.z, a3.z, q1.w, a3.w);
        let r14 = R::sum_prod4(q1.x, a4.x, q1.y, a4.y, q1.z, a4.z, q1.w, a4.w);
        let b2 = Qr4InternalTrait::project_out(a2, q1, r12);
        let b3 = Qr4InternalTrait::project_out(a3, q1, r13);
        let b4 = Qr4InternalTrait::project_out(a4, q1, r14);
        let r22 = R::norm4(b2.x, b2.y, b2.z, b2.w);
        let q2 = Qr4InternalTrait::unit(b2, r22);
        let r23 = R::sum_prod4(q2.x, b3.x, q2.y, b3.y, q2.z, b3.z, q2.w, b3.w);
        let r24 = R::sum_prod4(q2.x, b4.x, q2.y, b4.y, q2.z, b4.z, q2.w, b4.w);
        let c3 = Qr4InternalTrait::project_out(b3, q2, r23);
        let c4 = Qr4InternalTrait::project_out(b4, q2, r24);
        let r33 = R::norm4(c3.x, c3.y, c3.z, c3.w);
        let q3 = Qr4InternalTrait::unit(c3, r33);
        let r34 = R::sum_prod4(q3.x, c4.x, q3.y, c4.y, q3.z, c4.z, q3.w, c4.w);
        let d4 = Qr4InternalTrait::project_out(c4, q3, r34);
        let r44 = R::norm4(d4.x, d4.y, d4.z, d4.w);
        let q4 = Qr4InternalTrait::unit(d4, r44);
        Qr4 {
            q: Matrix4Trait::from_columns(q1, q2, q3, q4),
            r: Matrix4Trait::new(
                r11,
                r12,
                r13,
                r14,
                R::zero(),
                r22,
                r23,
                r24,
                R::zero(),
                R::zero(),
                r33,
                r34,
                R::zero(),
                R::zero(),
                R::zero(),
                r44,
            ),
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

    /// The upper triangular factor `R`, consuming the factorisation: `r()`. Exact. Upstream:
    /// `QR::unpack_r`.
    #[inline(always)]
    fn unpack_r(self: Qr4<T>) -> Matrix4<T> {
        self.r
    }

    /// The factors as the factorisation stores them, `(Q, R)`: modified Gram-Schmidt keeps both
    /// in full, where upstream's Householder storage packs the reflectors and `R` into one matrix
    /// (the representation is internal on both sides). Exact. Upstream: `QR::qr_internal`
    /// (`#[doc(hidden)]`).
    #[inline(always)]
    fn qr_internal(self: Qr4<T>) -> (Matrix4<T>, Matrix4<T>) {
        (self.q, self.r)
    }

    /// `rhs = Qᵀ * rhs` in place, for any `rhs` with 4 rows (a vector or a matrix): ONE fused sum
    /// of products per entry (`MatrixTrMul::tr_mul`, floored once). Upstream: `QR::q_tr_mul`,
    /// which applies the Householder reflections one after the other (a rounding each).
    fn q_tr_mul<B, impl K: SolveKernel<Matrix4<T>, B>, +Drop<B>>(self: Qr4<T>, ref rhs: B) {
        rhs = K::tr_mul_rhs(self.q, rhs);
    }

    /// Overwrites `b` (any shape with 4 rows: a vector or a matrix) with the solution of `A * x =
    /// b` and returns `true`, or returns `false` and leaves `b` unchanged when a diagonal entry of
    /// `R` is exactly zero (`is_invertible`; upstream returns `false` after overwriting `b` with
    /// `Qᵀ b`). `x = R⁻¹ (Qᵀ b)`: one fused `tr_mul`, then back substitution (one floor and
    /// one correctly rounded division per component, one prepared divisor per row from 3 columns
    /// on); on a vector it is bit-identical to `solve`. Panics on overflow. Upstream:
    /// `QR::solve_mut`.
    fn solve_mut<B, impl K: SolveKernel<Matrix4<T>, B>, +Drop<B>>(self: Qr4<T>, ref b: B) -> bool {
        if !Self::is_invertible(self) {
            return false;
        }
        b = K::upper(self.r, K::tr_mul_rhs(self.q, b));
        true
    }

    /// Whether the factorisation is invertible: all 4 diagonal entries of `R` are EXACTLY nonzero
    /// (no epsilon), like upstream's `QR::is_invertible`. Equivalently, `Q` is orthonormal.
    #[inline(always)]
    fn is_invertible(self: Qr4<T>) -> bool {
        self.r.m11 != R::zero()
            && self.r.m22 != R::zero()
            && self.r.m33 != R::zero()
            && self.r.m44 != R::zero()
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
        Some(Qr4InternalTrait::back_substitute(self, self.q.tr_mul(b)))
    }

    /// The inverse, or `None` when a diagonal entry of `R` is exactly zero (`is_invertible`).
    /// `A^-1 = R^-1 Qᵀ`, see `Qr3::try_inverse`. Upstream: `QR::try_inverse`.
    fn try_inverse(self: Qr4<T>) -> Option<Matrix4<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        Some(
            Matrix4Trait::from_columns(
                Qr4InternalTrait::back_substitute(self, self.q.row1()),
                Qr4InternalTrait::back_substitute(self, self.q.row2()),
                Qr4InternalTrait::back_substitute(self, self.q.row3()),
                Qr4InternalTrait::back_substitute(self, self.q.row4()),
            ),
        )
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
