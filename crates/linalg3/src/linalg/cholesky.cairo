//! Cholesky factorisation `A = L·Lᵀ` of a symmetric POSITIVE-DEFINITE matrix, unrolled for the
//! static sizes 2, 3, 4 and 6 (upstream `nalgebra::linalg::Cholesky`).
//!
//! `Cholesky{2,3,4,6}::new` returns `None` instead of a factor when the matrix is not positive
//! definite; the factorisation then offers `l`, `solve`, `inverse` and `determinant`, exactly the
//! upstream surface minus `rank_one_update` / `insert_column` / `remove_column` (see below).
//!
//! What is stored (DESIGN D4, "no loops, no zeros carried around"): ONLY the n(n+1)/2 components
//! of the lower triangle, as flat named fields `l11, l21, .., lnn` in column-major order. A full
//! `MatrixN` with explicit zeros would make `solve` read (and the Sierra code copy) n(n-1)/2
//! values that are known to be zero; `l()` materialises them on demand instead.
//!
//! Numeric contract (AGENTS.md rule 4): every sum of products — pivots, substitution dot
//! products, products of the triangular inverse — is accumulated EXACTLY in the `Real::Wide`
//! accumulator and floored ONCE. Divisions by a pivot are correctly rounded divisions, never a
//! multiplication by a rounded reciprocal, following the precedent of `Matrix3::try_inverse` and
//! `Vector3::unscale`.
//! The `alt_recip` candidates are kept in `benches.cairo` with their measurements. Since `fixed`
//! 0.3.0 (division rounded to nearest) they are the cheaper ones — 12 to 13 % in `solve`, 13 to
//! 22 % in `inverse` — and they still do not ship (WP 7.2): upstream's `solve_mut` (and
//! `inverse`, which is `solve_mut` on the identity) DIVIDES by each pivot, and `recip` + product
//! rounds twice where a division rounds once, which
//! `test_cholesky2_inverse_alt_recip_loses_low_bits` exhibits. Quotients of one row of `l⁻¹`
//! that share a pivot go through one prepared divisor (`Real::div3` .. `div5`, bit-identical to
//! per-element division).
//!
//! NOT ported: `rank_one_update` (rapier never updates a factor in place — it refactorises the
//! effective mass every step) and the dynamic `insert_column` / `remove_column`, which have no
//! meaning for a statically sized factor.
//!
//! When no square root is wanted, upstream's `UDU` (`crate::linalg::udu`, on the crate-internal
//! `LDLᵀ` kernel of DESIGN D6) factorises without one, indefinite symmetric matrices included.

use nalgebra_core::base::errors::NOT_POSITIVE_DEFINITE;
use nalgebra_core::internal::base::solve::SolveKernel;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::internal::base::sym_matrix3::SymMatrix3;
use simba::scalar::{Real, Transcendental};

/// The Cholesky factorisation `a = l * lᵀ` of a symmetric positive-definite 3x3 matrix:
/// the 6 components of the LOWER triangular factor `l`, `l33` last.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I >= J`); the entries above the diagonal
/// are zero and are not stored. Fields are declared column by column, so `Serde` writes
/// `l11, l21, .., l31, l22, ..` — the lower triangle in the column-major order upstream stores a
/// matrix in. Build one with `Cholesky3Trait::new`; the fields are public so that a factor
/// computed elsewhere can be re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Cholesky3<T> {
    /// Row 1, column 1 of the lower triangular factor.
    pub l11: T,
    /// Row 2, column 1 of the lower triangular factor.
    pub l21: T,
    /// Row 3, column 1 of the lower triangular factor.
    pub l31: T,
    /// Row 2, column 2 of the lower triangular factor.
    pub l22: T,
    /// Row 3, column 2 of the lower triangular factor.
    pub l32: T,
    /// Row 3, column 3 of the lower triangular factor.
    pub l33: T,
}

/// Methods of `Cholesky3<T>` for any `Real` scalar.
#[generate_trait]
pub impl Cholesky3Impl<
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
> of Cholesky3Trait<T> {
    /// The Cholesky factorisation `a = l * lᵀ` of the symmetric positive-definite `a`, or `None`
    /// when `a` is not positive definite. Upstream: `Cholesky::new`.
    ///
    /// Like upstream, only the LOWER triangle of `a` is read (the entries at row `i`,
    /// column `j` with `i >= j`): the strictly upper triangle is ignored and the symmetry
    /// of `a` is NOT checked.
    ///
    /// Column by column (j = 1..3): the pivot `p_j = a_jj - Σ_(k<j) l_jk²` is accumulated
    /// exactly in the wide accumulator and floored ONCE, then `l_jj = sqrt(p_j)` (exact floor of
    /// the square root); each sub-diagonal entry `l_ij = (a_ij - Σ_(k<j) l_ik·l_jk) / l_jj` costs
    /// one exact accumulation (one floor) and one correctly rounded division (a second rounding, to
    /// nearest).
    /// 3 square roots and 3 divisions in total.
    ///
    /// SINGULARITY CRITERION: `None` as soon as a pivot is `<= 0` AFTER that flooring. Upstream
    /// tests the real pivot against 0; flooring makes the test slightly stricter, so a matrix that
    /// is positive definite in exact arithmetic but whose j-th pivot is below 1 ulp (2^-32 in
    /// Q32.32) is rejected here — that pivot is indistinguishable from zero in the scalar and
    /// `sqrt` of it would carry no information. Indefinite or non-symmetric input is caught the
    /// same way, but only when it drives a pivot to zero: `new` is not a definiteness test.
    ///
    /// Panics on overflow of a pivot or a numerator; never wraps.
    #[inline(always)]
    fn new(a: Matrix3<T>) -> Option<Cholesky3<T>> {
        Cholesky3InternalTrait::new_sym(
            SymMatrix3 { m11: a.m11, m12: a.m21, m13: a.m31, m22: a.m22, m23: a.m32, m33: a.m33 },
        )
    }

    /// The lower triangular factor, with explicit zeros above the diagonal. Upstream:
    /// `Cholesky::l`. Exact: the stored components are copied, nothing is recomputed.
    #[inline(always)]
    fn l(self: Cholesky3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: self.l11,
            m21: self.l21,
            m31: self.l31,
            m12: R::zero(),
            m22: self.l22,
            m32: self.l32,
            m13: R::zero(),
            m23: R::zero(),
            m33: self.l33,
        }
    }

    /// The solution `x` of `a * x = b`, by forward substitution on `l` then back substitution on
    /// `lᵀ`. Upstream: `Cholesky::solve`.
    ///
    /// Each of the 6 intermediates is ONE exact accumulation floored once (`b_i - Σ_(k<i)
    /// l_ik·y_k`, then `y_i - Σ_(k>i) l_ki·x_k`) followed by one correctly rounded division
    /// by the pivot: two roundings per intermediate, 6 divisions in total. Never a multiplication
    /// by a rounded reciprocal (see the module doc).
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Cholesky3<T>, b: Vector3<T>) -> Vector3<T> {
        let y1 = R::div(b.x, self.l11);
        let w = R::wide_add(R::wide_zero(), b.y);
        let w = R::wide_sub_prod(w, self.l21, y1);
        let f2 = R::wide_rescale(w);
        let y2 = R::div(f2, self.l22);
        let w = R::wide_add(R::wide_zero(), b.z);
        let w = R::wide_sub_prod(w, self.l31, y1);
        let w = R::wide_sub_prod(w, self.l32, y2);
        let f3 = R::wide_rescale(w);
        let y3 = R::div(f3, self.l33);
        let x3 = R::div(y3, self.l33);
        let w = R::wide_add(R::wide_zero(), y2);
        let w = R::wide_sub_prod(w, self.l32, x3);
        let g2 = R::wide_rescale(w);
        let x2 = R::div(g2, self.l22);
        let w = R::wide_add(R::wide_zero(), y1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let w = R::wide_sub_prod(w, self.l31, x3);
        let g1 = R::wide_rescale(w);
        let x1 = R::div(g1, self.l11);
        Vector3 { x: x1, y: x2, z: x3 }
    }

    /// `a⁻¹ = l⁻ᵀ · l⁻¹`, as a full (symmetric) `Matrix3`, like upstream.
    /// Upstream: `Cholesky::inverse`.
    ///
    /// `q = l⁻¹` (lower triangular) is built column by column — `q_jj = recip(l_jj)` (the
    /// `1 / l_jj` rounded to nearest, cheaper than `ONE / l_jj`) and
    /// `q_ij = (-Σ_(k=j..i-1) l_ik·q_kj) / l_ii`, one exact accumulation and one correctly
    /// rounded division each — then `a⁻¹_ij = Σ_k q_ki·q_kj` is one exact accumulation
    /// floored once per output.
    /// 6 divisions in total, against the 18 of 3 `solve` calls on the columns
    /// of the identity.
    ///
    /// The result is symmetric by construction (one accumulation per unordered pair), so only its
    /// upper triangle is computed. Panics on overflow, which for an ill-conditioned `a` happens
    /// well before the mathematical inverse stops fitting in the scalar.
    fn inverse(self: Cholesky3<T>) -> Matrix3<T> {
        let q11 = R::recip(self.l11);
        let q22 = R::recip(self.l22);
        let q33 = R::recip(self.l33);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l21, q11);
        let t21 = R::wide_rescale(w);
        let q21 = R::div(t21, self.l22);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l31, q11);
        let w = R::wide_sub_prod(w, self.l32, q21);
        let t31 = R::wide_rescale(w);
        let q31 = R::div(t31, self.l33);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l32, q22);
        let t32 = R::wide_rescale(w);
        let q32 = R::div(t32, self.l33);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q11, q11);
        let w = R::wide_add_prod(w, q21, q21);
        let w = R::wide_add_prod(w, q31, q31);
        let r11 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q21, q22);
        let w = R::wide_add_prod(w, q31, q32);
        let r12 = R::wide_rescale(w);
        let r13 = q31 * q33;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q22, q22);
        let w = R::wide_add_prod(w, q32, q32);
        let r22 = R::wide_rescale(w);
        let r23 = q32 * q33;
        let r33 = R::sqr(q33);
        Matrix3 {
            m11: r11,
            m21: r12,
            m31: r13,
            m12: r12,
            m22: r22,
            m32: r23,
            m13: r13,
            m23: r23,
            m33: r33,
        }
    }

    /// `det(a) = Π l_jj²`, computed as `(Π l_jj)²`: a balanced product tree (2 floored
    /// products) then one `Real::sqr` (one more floor), i.e. 3 roundings instead of the
    /// 5 of `Π (l_jj²)`. Upstream: `Cholesky::determinant`.
    ///
    /// The intermediate `Π l_jj` is the square root of the result, so it overflows only if the
    /// determinant itself does — and then this panics. Underflow is silent (a determinant below
    /// 1 ulp floors to 0); that is the scalar's floor rounding, NOT a singularity report, since
    /// `new` already accepted the matrix.
    #[inline(always)]
    fn determinant(self: Cholesky3<T>) -> T {
        R::sqr(self.l11 * (self.l22 * self.l33))
    }

    /// The factorisation computed WITHOUT the positivity test, like upstream's
    /// `Cholesky::new_unchecked` — except that a pivot `<= 0` (upstream: the square root of a
    /// negative number or a division by zero, NaN / infinities, which a fixed-point scalar does
    /// not have) panics with `nalgebra: not positive definite`. Same kernel, same rounding and
    /// same gas as `new`. Upstream: `Cholesky::new_unchecked`.
    #[inline(always)]
    fn new_unchecked(matrix: Matrix3<T>) -> Cholesky3<T> {
        Self::new(matrix).expect(NOT_POSITIVE_DEFINITE)
    }

    /// `new` with `substitute` in place of every pivot that is `<= 0` after flooring (upstream:
    /// zero or negative, where the square root fails); `None` when such a pivot occurs and
    /// `substitute` is itself `<= 0`. The same kernel as `new` otherwise (same rounding). Upstream:
    /// `Cholesky::new_with_substitute`.
    #[inline(always)]
    fn new_with_substitute(a: Matrix3<T>, substitute: T) -> Option<Cholesky3<T>> {
        Cholesky3InternalTrait::new_sym_with_substitute(
            SymMatrix3 { m11: a.m11, m12: a.m21, m13: a.m31, m22: a.m22, m23: a.m32, m33: a.m33 },
            substitute,
        )
    }

    /// The factor `l` from the LOWER triangle of `matrix` (the strictly upper triangle is not
    /// read, like upstream's "dirty" storage). Nothing is checked. Exact: moves. Upstream:
    /// `Cholesky::pack_dirty`.
    #[inline(always)]
    fn pack_dirty(matrix: Matrix3<T>) -> Cholesky3<T> {
        Cholesky3 {
            l11: matrix.m11,
            l21: matrix.m21,
            l31: matrix.m31,
            l22: matrix.m22,
            l32: matrix.m32,
            l33: matrix.m33,
        }
    }

    /// The lower triangular factor, consuming the factorisation: `l()`. Exact. Upstream:
    /// `Cholesky::unpack`.
    #[inline(always)]
    fn unpack(self: Cholesky3<T>) -> Matrix3<T> {
        Self::l(self)
    }

    /// The factor with its "dirty" upper triangle: `l()` here, the compact storage keeps no upper
    /// triangle (upstream returns the input's upper triangle there, unspecified by its docs).
    /// Exact.
    /// Upstream: `Cholesky::unpack_dirty`.
    #[inline(always)]
    fn unpack_dirty(self: Cholesky3<T>) -> Matrix3<T> {
        Self::l(self)
    }

    /// The factor with its "dirty" upper triangle, by value: `l()` (see `unpack_dirty`). Exact.
    /// Upstream: `Cholesky::l_dirty` (a reference to the storage).
    #[inline(always)]
    fn l_dirty(self: Cholesky3<T>) -> Matrix3<T> {
        Self::l(self)
    }

    /// Overwrites `b` (any shape with 3 rows: a vector or a matrix) with the solution of `a * x =
    /// b`: `l y = b` by forward substitution then `lᵀ x = y` by back substitution, each component
    /// ONE exact sum floored once then one correctly rounded division by the pivot (one prepared
    /// divisor per row from 3 columns on). On a vector it is bit-identical to `solve`. Panics on
    /// overflow. Upstream: `Cholesky::solve_mut`.
    fn solve_mut<B, impl K: SolveKernel<Matrix3<T>, B>, +Drop<B>>(self: Cholesky3<T>, ref b: B) {
        let lt = Matrix3 {
            m11: self.l11,
            m21: R::zero(),
            m31: R::zero(),
            m12: self.l21,
            m22: self.l22,
            m32: R::zero(),
            m13: self.l31,
            m23: self.l32,
            m33: self.l33,
        };
        b = K::upper(lt, K::lower(Self::l(self), b));
    }

    /// `ln(det(a)) = Σ ln(l_jj²)`, computed as `2 Σ ln(l_jj)`: 3 natural logarithms, an exact
    /// sum and an exact doubling. Upstream sums `ln(l_jj²)`; squaring first would floor `l_jj²`
    /// (and send a pivot below 2^-16 to `ln(0)`), so the logarithm of the pivot itself is both
    /// cheaper and more accurate. Needs `Transcendental`. Upstream: `Cholesky::ln_determinant`.
    fn ln_determinant<+Transcendental<T>>(self: Cholesky3<T>) -> T {
        let s = Transcendental::ln(self.l11)
            + Transcendental::ln(self.l22)
            + Transcendental::ln(self.l33);
        s + s
    }
}

/// Crate-internal kernel of `Cholesky3<T>`: `new` on the 6 independent components of the
/// symmetric matrix (the public `new` takes upstream's full `Matrix3` and is inlined around it, so
/// the call passes 6 scalars instead of 9: WP 8.0 keeps the gas of the former `SymMatrix3`
/// signature).
#[generate_trait]
pub(crate) impl Cholesky3InternalImpl<
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
> of Cholesky3InternalTrait<T> {
    /// The factorisation of the symmetric matrix of upper triangle `a`; see `Cholesky3Trait::new`.
    fn new_sym(a: SymMatrix3<T>) -> Option<Cholesky3<T>> {
        let p1 = a.m11;
        if p1 <= R::zero() {
            return None;
        }
        let l11 = R::sqrt(p1);
        let l21 = R::div(a.m12, l11);
        let l31 = R::div(a.m13, l11);
        let w = R::wide_add(R::wide_zero(), a.m22);
        let w = R::wide_sub_prod(w, l21, l21);
        let p2 = R::wide_rescale(w);
        if p2 <= R::zero() {
            return None;
        }
        let l22 = R::sqrt(p2);
        let w = R::wide_add(R::wide_zero(), a.m23);
        let w = R::wide_sub_prod(w, l31, l21);
        let n32 = R::wide_rescale(w);
        let l32 = R::div(n32, l22);
        let w = R::wide_add(R::wide_zero(), a.m33);
        let w = R::wide_sub_prod(w, l31, l31);
        let w = R::wide_sub_prod(w, l32, l32);
        let p3 = R::wide_rescale(w);
        if p3 <= R::zero() {
            return None;
        }
        let l33 = R::sqrt(p3);
        Some(Cholesky3 { l11, l21, l31, l22, l32, l33 })
    }

    /// `new_sym` with `substitute` in place of every pivot that is `<= 0` after flooring (still
    /// `None` if `substitute` itself is `<= 0`); see `Cholesky3Trait::new_with_substitute`.
    fn new_sym_with_substitute(a: SymMatrix3<T>, substitute: T) -> Option<Cholesky3<T>> {
        let p1 = a.m11;
        let p1 = if p1 <= R::zero() {
            substitute
        } else {
            p1
        };
        if p1 <= R::zero() {
            return None;
        }
        let l11 = R::sqrt(p1);
        let l21 = R::div(a.m12, l11);
        let l31 = R::div(a.m13, l11);
        let w = R::wide_add(R::wide_zero(), a.m22);
        let w = R::wide_sub_prod(w, l21, l21);
        let p2 = R::wide_rescale(w);
        let p2 = if p2 <= R::zero() {
            substitute
        } else {
            p2
        };
        if p2 <= R::zero() {
            return None;
        }
        let l22 = R::sqrt(p2);
        let w = R::wide_add(R::wide_zero(), a.m23);
        let w = R::wide_sub_prod(w, l31, l21);
        let n32 = R::wide_rescale(w);
        let l32 = R::div(n32, l22);
        let w = R::wide_add(R::wide_zero(), a.m33);
        let w = R::wide_sub_prod(w, l31, l31);
        let w = R::wide_sub_prod(w, l32, l32);
        let p3 = R::wide_rescale(w);
        let p3 = if p3 <= R::zero() {
            substitute
        } else {
            p3
        };
        if p3 <= R::zero() {
            return None;
        }
        let l33 = R::sqrt(p3);
        Some(Cholesky3 { l11, l21, l31, l22, l32, l33 })
    }
}

/// `SquareMatrix::cholesky` on `Matrix3<T>` (upstream `nalgebra::linalg` decomposition entry
/// point).
#[generate_trait]
pub impl Matrix3CholeskyImpl<
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
> of Matrix3CholeskyTrait<T> {
    /// The Cholesky factorisation of `self` (its lower triangle), or `None` when it is not
    /// positive definite: `Cholesky3Trait::new(self)`. Upstream: `SquareMatrix::cholesky`.
    #[inline(always)]
    fn cholesky(self: Matrix3<T>) -> Option<Cholesky3<T>> {
        Cholesky3Trait::new(self)
    }
}
