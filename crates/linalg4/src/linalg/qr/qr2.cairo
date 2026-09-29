//! `Qr2`: the QR factorisation of a `Matrix2` (upstream `nalgebra::linalg::QR` on a 2x2 matrix).
//!
//! `A = Q * R` with `Q` orthonormal and `R` upper triangular with a non-negative diagonal — the
//! convention of upstream's unpacked `q()` / `r()`, see the module doc of `linalg::qr`.

use nalgebra_core::base::matrix2::Matrix2;
use nalgebra_core::base::matrix_tr_mul::MatrixTrMul;
use nalgebra_core::base::vector2::Vector2;
use nalgebra_core::internal::base::solve::SolveKernel;
use nalgebra_static3::base::matrix2::Matrix2Trait;
use simba::scalar::Real;

/// The QR factorisation of a `Matrix2<T>`: `A = Q * R`.
///
/// `q` is orthonormal (a rotation or a reflection) and `r` is upper triangular with
/// `r_ii >= 0` and an exactly zero strict lower triangle. Built by `Qr2Trait::new` or
/// `Matrix2QrTrait::qr`. Upstream: `nalgebra::linalg::QR`, which packs the Householder reflectors
/// and the signed diagonal instead and rebuilds the two factors on demand.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Qr2<T> {
    /// The orthonormal factor `Q` (orthonormal iff `is_invertible`, see the module doc).
    pub q: Matrix2<T>,
    /// The upper triangular factor `R`, `r_ii >= 0`.
    pub r: Matrix2<T>,
}

/// Methods of `Qr2<T>` for any `Real` scalar.
#[generate_trait]
pub impl Qr2Impl<
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
> of Qr2Trait<T> {
    /// The QR factorisation of `matrix` by modified Gram-Schmidt, fully unrolled. Upstream:
    /// `matrix.qr()` / `QR::new(matrix)` (Householder, same unpacked factors).
    ///
    /// ```text
    /// r11 = |a1|            q1 = a1 / r11
    /// r12 = <q1, a2>        w  = a2 - r12 q1
    /// r22 = |w|             q2 = w / r22
    /// ```
    ///
    /// Always succeeds. `r11` and `r22` are floored `norm2`, whose sum of squares is accumulated
    /// unscaled, so no intermediate can overflow and each norm is the exact floor of the true one;
    /// `r12` is one fused `sum_prod2`, `w` two fused `mul_add`, and each component of `q1` / `q2`
    /// one correctly rounded division. Five roundings per column.
    ///
    /// A column that is exactly zero leaves `r_ii = 0` and sets `q_i = 0` rather than completing
    /// the basis (module doc): `Q * R = A` still holds exactly, and `is_invertible` is false.
    ///
    /// **Measured** on the 30 vectors of the `qr2_q_r` oracle suite: the factors agree with
    /// upstream's unpacked Householder factors within **25 ulp** (`Q`) and **78 ulp** (`R`),
    /// every case inside the oracle tolerance; `|A - Q R| <= 2 ulp * max(1, max |a_ij|)` and
    /// `|QᵀQ - I| <= 54 ulp` (dominated by the cases whose `r22` is small — the suite
    /// goes down to 0.036 — since `q_2 = w / r22` inherits the rounding of `w` divided by it).
    ///
    /// Panics with the scalar's overflow error if a norm does not fit (a component of `q` cannot:
    /// it is bounded by 1).
    fn new(matrix: Matrix2<T>) -> Qr2<T> {
        let r11 = R::norm2(matrix.m11, matrix.m21);
        let (q11, q21) = if r11 == R::zero() {
            (R::zero(), R::zero())
        } else {
            (R::div(matrix.m11, r11), R::div(matrix.m21, r11))
        };
        let r12 = R::sum_prod2(q11, matrix.m12, q21, matrix.m22);
        let w1 = R::mul_add(-r12, q11, matrix.m12);
        let w2 = R::mul_add(-r12, q21, matrix.m22);
        let r22 = R::norm2(w1, w2);
        let (q12, q22) = if r22 == R::zero() {
            (R::zero(), R::zero())
        } else {
            (R::div(w1, r22), R::div(w2, r22))
        };
        Qr2 {
            q: Matrix2 { m11: q11, m21: q21, m12: q12, m22: q22 },
            r: Matrix2 { m11: r11, m21: R::zero(), m12: r12, m22: r22 },
        }
    }

    /// The orthonormal factor `Q`. Exact: a move. Upstream: `QR::q`, which rebuilds it from the
    /// stored reflectors.
    #[inline(always)]
    fn q(self: Qr2<T>) -> Matrix2<T> {
        self.q
    }

    /// The upper triangular factor `R`. Exact: a move. Upstream: `QR::r`.
    #[inline(always)]
    fn r(self: Qr2<T>) -> Matrix2<T> {
        self.r
    }

    /// `(Q, R)`. Exact: moves. Upstream: `QR::unpack`.
    #[inline(always)]
    fn unpack(self: Qr2<T>) -> (Matrix2<T>, Matrix2<T>) {
        (self.q, self.r)
    }

    /// The upper triangular factor `R`, consuming the factorisation: `r()`. Exact. Upstream:
    /// `QR::unpack_r`.
    #[inline(always)]
    fn unpack_r(self: Qr2<T>) -> Matrix2<T> {
        self.r
    }

    /// The factors as the factorisation stores them, `(Q, R)`: modified Gram-Schmidt keeps both
    /// in full, where upstream's Householder storage packs the reflectors and `R` into one matrix
    /// (the representation is internal on both sides). Exact. Upstream: `QR::qr_internal`
    /// (`#[doc(hidden)]`).
    #[inline(always)]
    fn qr_internal(self: Qr2<T>) -> (Matrix2<T>, Matrix2<T>) {
        (self.q, self.r)
    }

    /// `rhs = Qᵀ * rhs` in place, for any `rhs` with 2 rows (a vector or a matrix): ONE fused sum
    /// of products per entry (`MatrixTrMul::tr_mul`, floored once). Upstream: `QR::q_tr_mul`,
    /// which applies the Householder reflections one after the other (a rounding each).
    fn q_tr_mul<B, impl K: SolveKernel<Matrix2<T>, B>, +Drop<B>>(self: Qr2<T>, ref rhs: B) {
        rhs = K::tr_mul_rhs(self.q, rhs);
    }

    /// Overwrites `b` (any shape with 2 rows: a vector or a matrix) with the solution of `A * x =
    /// b` and returns `true`, or returns `false` and leaves `b` unchanged when a diagonal entry of
    /// `R` is exactly zero (`is_invertible`; upstream returns `false` after overwriting `b` with
    /// `Qᵀ b`). `x = R⁻¹ (Qᵀ b)`: one fused `tr_mul`, then back substitution (one floor and
    /// one correctly rounded division per component, one prepared divisor per row from 3 columns
    /// on); on a vector it is bit-identical to `solve`. Panics on overflow. Upstream:
    /// `QR::solve_mut`.
    fn solve_mut<B, impl K: SolveKernel<Matrix2<T>, B>, +Drop<B>>(self: Qr2<T>, ref b: B) -> bool {
        if !Self::is_invertible(self) {
            return false;
        }
        b = K::upper(self.r, K::tr_mul_rhs(self.q, b));
        true
    }

    /// Whether the factorisation is invertible: all 2 diagonal entries of `R` are EXACTLY nonzero
    /// (no epsilon), like upstream's `QR::is_invertible`. Equivalently, `Q` is orthonormal.
    #[inline(always)]
    fn is_invertible(self: Qr2<T>) -> bool {
        self.r.m11 != R::zero() && self.r.m22 != R::zero()
    }

    /// The solution of `A * x = b` for the factored `A`, or `None` when a diagonal entry of `R`
    /// is exactly zero (`is_invertible`). Upstream: `QR::solve`.
    ///
    /// `x = R^-1 (Qᵀ b)`: one fused `tr_mul` for `Qᵀ b` (one rounding per component), then
    /// back substitution, each component costing one fused numerator and one correctly rounded
    /// division.
    /// Panics with the scalar's overflow error if a component of `x` does not fit.
    fn solve(self: Qr2<T>, b: Vector2<T>) -> Option<Vector2<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        let y = self.q.tr_mul(b);
        let x2 = R::div(y.y, self.r.m22);
        let x1 = R::div(R::mul_add(-self.r.m12, x2, y.x), self.r.m11);
        Some(Vector2 { x: x1, y: x2 })
    }

    /// The inverse, or `None` when a diagonal entry of `R` is exactly zero (`is_invertible`).
    /// Upstream: `QR::try_inverse`.
    ///
    /// `A^-1 = R^-1 Qᵀ`: the transpose is free (moves), and the back substitution runs on the two
    /// columns of `Qᵀ` at once. Two roundings per entry (the fused numerator, then the floor
    /// division by the pivot); a reciprocal per pivot would amortise over the 2 columns and is not
    /// used, for the reason `Lu2::try_inverse` documents (a second rounding per output scalar).
    /// Panics with the scalar's overflow error if an entry does not fit.
    fn try_inverse(self: Qr2<T>) -> Option<Matrix2<T>> {
        if !Self::is_invertible(self) {
            return None;
        }
        // Column j of the result solves `R x = (Qᵀ)_j`, and `(Qᵀ)_j` is row j of `Q`.
        let x21 = R::div(self.q.m12, self.r.m22);
        let x22 = R::div(self.q.m22, self.r.m22);
        let x11 = R::div(R::mul_add(-self.r.m12, x21, self.q.m11), self.r.m11);
        let x12 = R::div(R::mul_add(-self.r.m12, x22, self.q.m21), self.r.m11);
        Some(Matrix2 { m11: x11, m21: x21, m12: x12, m22: x22 })
    }
}

/// Crate-internal kernels of `Qr2<T>` (WP 8.0: the public API is strictly upstream's).
/// `determinant`
/// has no upstream counterpart (upstream's `QR::determinant` is commented out); the tests use it to
/// check the factors.
#[generate_trait]
pub(crate) impl Qr2InternalImpl<
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
> of Qr2InternalTrait<T> {
    /// The determinant: `det(Q) * r11 * r22`. Upstream: `QR::determinant`.
    ///
    /// `R` has a non-negative diagonal, so the sign lives entirely in `det(Q) = ±1`, read off
    /// `Matrix2::determinant(q)` — a single `diff_prod` on an orthonormal matrix, hence `±1`
    /// within the rounding of the factorisation, and its SIGN is what is used. The sign is applied
    /// to the FIRST factor (an exact negation) so that the product stays a floor chain, exactly as
    /// `Lu2::determinant` does.
    ///
    /// Exactly zero for a rank-deficient matrix. Panics with the scalar's overflow error if the
    /// product does not fit.
    ///
    /// PREFER `Matrix2::determinant` when the factorisation is not needed for something else: the
    /// closed form is one exactly-rounded `diff_prod`, where this one multiplies two norms that
    /// already carry the rounding of the orthogonalisation.
    fn determinant(self: Qr2<T>) -> T {
        let d = if self.q.determinant().is_sign_negative() {
            -self.r.m11
        } else {
            self.r.m11
        };
        d * self.r.m22
    }
}

/// `Matrix2` methods that go through the QR factorisation; upstream carries them on the matrix
/// itself. Import `Matrix2QrTrait` to use them.
#[generate_trait]
pub impl Matrix2QrImpl<
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
> of Matrix2QrTrait<T> {
    /// The QR factorisation. Upstream: `Matrix2::qr`.
    #[inline(always)]
    fn qr(self: Matrix2<T>) -> Qr2<T> {
        Qr2Trait::new(self)
    }
}
