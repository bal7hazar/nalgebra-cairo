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

use simba::scalar::Real;
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::matrix4::Matrix4;
use crate::base::matrix6::Matrix6;
use crate::base::sym_matrix2::SymMatrix2;
use crate::base::sym_matrix3::SymMatrix3;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector6::Vector6;

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;

/// The Cholesky factorisation `a = l * lᵀ` of a symmetric positive-definite 2x2 matrix:
/// the 3 components of the LOWER triangular factor `l`, `l22` last.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I >= J`); the entries above the diagonal
/// are zero and are not stored. Fields are declared column by column, so `Serde` writes
/// `l11, l21, .., l21, l22, ..` — the lower triangle in the column-major order upstream stores a
/// matrix in. Build one with `Cholesky2Trait::new`; the fields are public so that a factor
/// computed elsewhere can be re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Cholesky2<T> {
    /// Row 1, column 1 of the lower triangular factor.
    pub l11: T,
    /// Row 2, column 1 of the lower triangular factor.
    pub l21: T,
    /// Row 2, column 2 of the lower triangular factor.
    pub l22: T,
}

/// Test-only field-wise equality (upstream `Cholesky2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Cholesky2PartialEq<T, +PartialEq<T>> of PartialEq<Cholesky2<T>> {
    fn eq(lhs: @Cholesky2<T>, rhs: @Cholesky2<T>) -> bool {
        lhs.l11 == rhs.l11 && lhs.l21 == rhs.l21 && lhs.l22 == rhs.l22
    }
}

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

/// Test-only field-wise equality (upstream `Cholesky3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Cholesky3PartialEq<T, +PartialEq<T>> of PartialEq<Cholesky3<T>> {
    fn eq(lhs: @Cholesky3<T>, rhs: @Cholesky3<T>) -> bool {
        lhs.l11 == rhs.l11
            && lhs.l21 == rhs.l21
            && lhs.l31 == rhs.l31
            && lhs.l22 == rhs.l22
            && lhs.l32 == rhs.l32
            && lhs.l33 == rhs.l33
    }
}

/// The Cholesky factorisation `a = l * lᵀ` of a symmetric positive-definite 4x4 matrix:
/// the 10 components of the LOWER triangular factor `l`, `l44` last.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I >= J`); the entries above the diagonal
/// are zero and are not stored. Fields are declared column by column, so `Serde` writes
/// `l11, l21, .., l41, l22, ..` — the lower triangle in the column-major order upstream stores a
/// matrix in. Build one with `Cholesky4Trait::new`; the fields are public so that a factor
/// computed elsewhere can be re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Cholesky4<T> {
    /// Row 1, column 1 of the lower triangular factor.
    pub l11: T,
    /// Row 2, column 1 of the lower triangular factor.
    pub l21: T,
    /// Row 3, column 1 of the lower triangular factor.
    pub l31: T,
    /// Row 4, column 1 of the lower triangular factor.
    pub l41: T,
    /// Row 2, column 2 of the lower triangular factor.
    pub l22: T,
    /// Row 3, column 2 of the lower triangular factor.
    pub l32: T,
    /// Row 4, column 2 of the lower triangular factor.
    pub l42: T,
    /// Row 3, column 3 of the lower triangular factor.
    pub l33: T,
    /// Row 4, column 3 of the lower triangular factor.
    pub l43: T,
    /// Row 4, column 4 of the lower triangular factor.
    pub l44: T,
}

/// Test-only field-wise equality (upstream `Cholesky4` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Cholesky4PartialEq<T, +PartialEq<T>> of PartialEq<Cholesky4<T>> {
    fn eq(lhs: @Cholesky4<T>, rhs: @Cholesky4<T>) -> bool {
        lhs.l11 == rhs.l11
            && lhs.l21 == rhs.l21
            && lhs.l31 == rhs.l31
            && lhs.l41 == rhs.l41
            && lhs.l22 == rhs.l22
            && lhs.l32 == rhs.l32
            && lhs.l42 == rhs.l42
            && lhs.l33 == rhs.l33
            && lhs.l43 == rhs.l43
            && lhs.l44 == rhs.l44
    }
}

/// The Cholesky factorisation `a = l * lᵀ` of a symmetric positive-definite 6x6 matrix:
/// the 21 components of the LOWER triangular factor `l`, `l66` last.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I >= J`); the entries above the diagonal
/// are zero and are not stored. Fields are declared column by column, so `Serde` writes
/// `l11, l21, .., l61, l22, ..` — the lower triangle in the column-major order upstream stores a
/// matrix in. Build one with `Cholesky6Trait::new`; the fields are public so that a factor
/// computed elsewhere can be re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Cholesky6<T> {
    /// Row 1, column 1 of the lower triangular factor.
    pub l11: T,
    /// Row 2, column 1 of the lower triangular factor.
    pub l21: T,
    /// Row 3, column 1 of the lower triangular factor.
    pub l31: T,
    /// Row 4, column 1 of the lower triangular factor.
    pub l41: T,
    /// Row 5, column 1 of the lower triangular factor.
    pub l51: T,
    /// Row 6, column 1 of the lower triangular factor.
    pub l61: T,
    /// Row 2, column 2 of the lower triangular factor.
    pub l22: T,
    /// Row 3, column 2 of the lower triangular factor.
    pub l32: T,
    /// Row 4, column 2 of the lower triangular factor.
    pub l42: T,
    /// Row 5, column 2 of the lower triangular factor.
    pub l52: T,
    /// Row 6, column 2 of the lower triangular factor.
    pub l62: T,
    /// Row 3, column 3 of the lower triangular factor.
    pub l33: T,
    /// Row 4, column 3 of the lower triangular factor.
    pub l43: T,
    /// Row 5, column 3 of the lower triangular factor.
    pub l53: T,
    /// Row 6, column 3 of the lower triangular factor.
    pub l63: T,
    /// Row 4, column 4 of the lower triangular factor.
    pub l44: T,
    /// Row 5, column 4 of the lower triangular factor.
    pub l54: T,
    /// Row 6, column 4 of the lower triangular factor.
    pub l64: T,
    /// Row 5, column 5 of the lower triangular factor.
    pub l55: T,
    /// Row 6, column 5 of the lower triangular factor.
    pub l65: T,
    /// Row 6, column 6 of the lower triangular factor.
    pub l66: T,
}

/// Test-only field-wise equality (upstream `Cholesky6` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Cholesky6PartialEq<T, +PartialEq<T>> of PartialEq<Cholesky6<T>> {
    fn eq(lhs: @Cholesky6<T>, rhs: @Cholesky6<T>) -> bool {
        lhs.l11 == rhs.l11
            && lhs.l21 == rhs.l21
            && lhs.l31 == rhs.l31
            && lhs.l41 == rhs.l41
            && lhs.l51 == rhs.l51
            && lhs.l61 == rhs.l61
            && lhs.l22 == rhs.l22
            && lhs.l32 == rhs.l32
            && lhs.l42 == rhs.l42
            && lhs.l52 == rhs.l52
            && lhs.l62 == rhs.l62
            && lhs.l33 == rhs.l33
            && lhs.l43 == rhs.l43
            && lhs.l53 == rhs.l53
            && lhs.l63 == rhs.l63
            && lhs.l44 == rhs.l44
            && lhs.l54 == rhs.l54
            && lhs.l64 == rhs.l64
            && lhs.l55 == rhs.l55
            && lhs.l65 == rhs.l65
            && lhs.l66 == rhs.l66
    }
}

/// Methods of `Cholesky2<T>` for any `Real` scalar.
#[generate_trait]
pub impl Cholesky2Impl<
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
> of Cholesky2Trait<T> {
    /// The Cholesky factorisation `a = l * lᵀ` of the symmetric positive-definite `a`, or `None`
    /// when `a` is not positive definite. Upstream: `Cholesky::new`.
    ///
    /// Like upstream, only the LOWER triangle of `a` is read (the entries at row `i`,
    /// column `j` with `i >= j`): the strictly upper triangle is ignored and the symmetry
    /// of `a` is NOT checked.
    ///
    /// Column by column (j = 1..2): the pivot `p_j = a_jj - Σ_(k<j) l_jk²` is accumulated
    /// exactly in the wide accumulator and floored ONCE, then `l_jj = sqrt(p_j)` (exact floor of
    /// the square root); each sub-diagonal entry `l_ij = (a_ij - Σ_(k<j) l_ik·l_jk) / l_jj` costs
    /// one exact accumulation (one floor) and one correctly rounded division (a second rounding, to
    /// nearest).
    /// 2 square roots and 1 division in total.
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
    fn new(a: Matrix2<T>) -> Option<Cholesky2<T>> {
        Cholesky2InternalTrait::new_sym(SymMatrix2 { m11: a.m11, m12: a.m21, m22: a.m22 })
    }

    /// The lower triangular factor, with explicit zeros above the diagonal. Upstream:
    /// `Cholesky::l`. Exact: the stored components are copied, nothing is recomputed.
    #[inline(always)]
    fn l(self: Cholesky2<T>) -> Matrix2<T> {
        Matrix2 { m11: self.l11, m21: self.l21, m12: R::zero(), m22: self.l22 }
    }

    /// The solution `x` of `a * x = b`, by forward substitution on `l` then back substitution on
    /// `lᵀ`. Upstream: `Cholesky::solve`.
    ///
    /// Each of the 4 intermediates is ONE exact accumulation floored once (`b_i - Σ_(k<i)
    /// l_ik·y_k`, then `y_i - Σ_(k>i) l_ki·x_k`) followed by one correctly rounded division
    /// by the pivot: two roundings per intermediate, 4 divisions in total. Never a multiplication
    /// by a rounded reciprocal (see the module doc).
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Cholesky2<T>, b: Vector2<T>) -> Vector2<T> {
        let y1 = R::div(b.x, self.l11);
        let w = R::wide_add(R::wide_zero(), b.y);
        let w = R::wide_sub_prod(w, self.l21, y1);
        let f2 = R::wide_rescale(w);
        let y2 = R::div(f2, self.l22);
        let x2 = R::div(y2, self.l22);
        let w = R::wide_add(R::wide_zero(), y1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let g1 = R::wide_rescale(w);
        let x1 = R::div(g1, self.l11);
        Vector2 { x: x1, y: x2 }
    }

    /// `a⁻¹ = l⁻ᵀ · l⁻¹`, as a full (symmetric) `Matrix2`, like upstream.
    /// Upstream: `Cholesky::inverse`.
    ///
    /// `q = l⁻¹` (lower triangular) is built column by column — `q_jj = recip(l_jj)` (the
    /// `1 / l_jj` rounded to nearest, cheaper than `ONE / l_jj`) and
    /// `q_ij = (-Σ_(k=j..i-1) l_ik·q_kj) / l_ii`, one exact accumulation and one correctly
    /// rounded division each — then `a⁻¹_ij = Σ_k q_ki·q_kj` is one exact accumulation
    /// floored once per output.
    /// 3 divisions in total, against the 8 of 2 `solve` calls on the columns
    /// of the identity.
    ///
    /// The result is symmetric by construction (one accumulation per unordered pair), so only its
    /// upper triangle is computed. Panics on overflow, which for an ill-conditioned `a` happens
    /// well before the mathematical inverse stops fitting in the scalar.
    fn inverse(self: Cholesky2<T>) -> Matrix2<T> {
        let q11 = R::recip(self.l11);
        let q22 = R::recip(self.l22);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l21, q11);
        let t21 = R::wide_rescale(w);
        let q21 = R::div(t21, self.l22);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q11, q11);
        let w = R::wide_add_prod(w, q21, q21);
        let r11 = R::wide_rescale(w);
        let r12 = q21 * q22;
        let r22 = R::sqr(q22);
        Matrix2 { m11: r11, m21: r12, m12: r12, m22: r22 }
    }

    /// `det(a) = Π l_jj²`, computed as `(Π l_jj)²`: a balanced product tree (1 floored
    /// product) then one `Real::sqr` (one more floor), i.e. 2 roundings instead of the
    /// 3 of `Π (l_jj²)`. Upstream: `Cholesky::determinant`.
    ///
    /// The intermediate `Π l_jj` is the square root of the result, so it overflows only if the
    /// determinant itself does — and then this panics. Underflow is silent (a determinant below
    /// 1 ulp floors to 0); that is the scalar's floor rounding, NOT a singularity report, since
    /// `new` already accepted the matrix.
    #[inline(always)]
    fn determinant(self: Cholesky2<T>) -> T {
        R::sqr(self.l11 * self.l22)
    }
}

/// Crate-internal kernel of `Cholesky2<T>`: `new` on the 3 independent components of the
/// symmetric matrix (the public `new` takes upstream's full `Matrix2` and is inlined around it, so
/// the call passes 3 scalars instead of 4: WP 8.0 keeps the gas of the former `SymMatrix2`
/// signature).
#[generate_trait]
pub(crate) impl Cholesky2InternalImpl<
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
> of Cholesky2InternalTrait<T> {
    /// The factorisation of the symmetric matrix of upper triangle `a`; see `Cholesky2Trait::new`.
    fn new_sym(a: SymMatrix2<T>) -> Option<Cholesky2<T>> {
        let p1 = a.m11;
        if p1 <= R::zero() {
            return None;
        }
        let l11 = R::sqrt(p1);
        let l21 = R::div(a.m12, l11);
        let w = R::wide_add(R::wide_zero(), a.m22);
        let w = R::wide_sub_prod(w, l21, l21);
        let p2 = R::wide_rescale(w);
        if p2 <= R::zero() {
            return None;
        }
        let l22 = R::sqrt(p2);
        Some(Cholesky2 { l11, l21, l22 })
    }
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
}

/// Methods of `Cholesky4<T>` for any `Real` scalar.
#[generate_trait]
pub impl Cholesky4Impl<
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
> of Cholesky4Trait<T> {
    /// The Cholesky factorisation `a = l * lᵀ` of the symmetric positive-definite `a`, or `None`
    /// when `a` is not positive definite. Upstream: `Cholesky::new`.
    ///
    /// Like upstream, only the LOWER triangle of `a` is read (the entries at row `i`,
    /// column `j` with `i >= j`): the strictly upper triangle is ignored and the symmetry
    /// of `a` is NOT checked.
    ///
    /// Column by column (j = 1..4): the pivot `p_j = a_jj - Σ_(k<j) l_jk²` is accumulated
    /// exactly in the wide accumulator and floored ONCE, then `l_jj = sqrt(p_j)` (exact floor of
    /// the square root); each sub-diagonal entry `l_ij = (a_ij - Σ_(k<j) l_ik·l_jk) / l_jj` costs
    /// one exact accumulation (one floor) and one correctly rounded division (a second rounding, to
    /// nearest).
    /// 4 square roots and 6 divisions in total.
    ///
    /// SINGULARITY CRITERION: `None` as soon as a pivot is `<= 0` AFTER that flooring. Upstream
    /// tests the real pivot against 0; flooring makes the test slightly stricter, so a matrix that
    /// is positive definite in exact arithmetic but whose j-th pivot is below 1 ulp (2^-32 in
    /// Q32.32) is rejected here — that pivot is indistinguishable from zero in the scalar and
    /// `sqrt` of it would carry no information. Indefinite or non-symmetric input is caught the
    /// same way, but only when it drives a pivot to zero: `new` is not a definiteness test.
    ///
    /// Panics on overflow of a pivot or a numerator; never wraps.
    fn new(a: Matrix4<T>) -> Option<Cholesky4<T>> {
        let p1 = a.m11;
        if p1 <= R::zero() {
            return None;
        }
        let l11 = R::sqrt(p1);
        let (l21, l31, l41) = R::div3(a.m21, a.m31, a.m41, l11);
        let w = R::wide_add(R::wide_zero(), a.m22);
        let w = R::wide_sub_prod(w, l21, l21);
        let p2 = R::wide_rescale(w);
        if p2 <= R::zero() {
            return None;
        }
        let l22 = R::sqrt(p2);
        let w = R::wide_add(R::wide_zero(), a.m32);
        let w = R::wide_sub_prod(w, l31, l21);
        let n32 = R::wide_rescale(w);
        let l32 = R::div(n32, l22);
        let w = R::wide_add(R::wide_zero(), a.m42);
        let w = R::wide_sub_prod(w, l41, l21);
        let n42 = R::wide_rescale(w);
        let l42 = R::div(n42, l22);
        let w = R::wide_add(R::wide_zero(), a.m33);
        let w = R::wide_sub_prod(w, l31, l31);
        let w = R::wide_sub_prod(w, l32, l32);
        let p3 = R::wide_rescale(w);
        if p3 <= R::zero() {
            return None;
        }
        let l33 = R::sqrt(p3);
        let w = R::wide_add(R::wide_zero(), a.m43);
        let w = R::wide_sub_prod(w, l41, l31);
        let w = R::wide_sub_prod(w, l42, l32);
        let n43 = R::wide_rescale(w);
        let l43 = R::div(n43, l33);
        let w = R::wide_add(R::wide_zero(), a.m44);
        let w = R::wide_sub_prod(w, l41, l41);
        let w = R::wide_sub_prod(w, l42, l42);
        let w = R::wide_sub_prod(w, l43, l43);
        let p4 = R::wide_rescale(w);
        if p4 <= R::zero() {
            return None;
        }
        let l44 = R::sqrt(p4);
        Some(Cholesky4 { l11, l21, l31, l41, l22, l32, l42, l33, l43, l44 })
    }

    /// The lower triangular factor, with explicit zeros above the diagonal. Upstream:
    /// `Cholesky::l`. Exact: the stored components are copied, nothing is recomputed.
    #[inline(always)]
    fn l(self: Cholesky4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: self.l11,
            m21: self.l21,
            m31: self.l31,
            m41: self.l41,
            m12: R::zero(),
            m22: self.l22,
            m32: self.l32,
            m42: self.l42,
            m13: R::zero(),
            m23: R::zero(),
            m33: self.l33,
            m43: self.l43,
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m44: self.l44,
        }
    }

    /// The solution `x` of `a * x = b`, by forward substitution on `l` then back substitution on
    /// `lᵀ`. Upstream: `Cholesky::solve`.
    ///
    /// Each of the 8 intermediates is ONE exact accumulation floored once (`b_i - Σ_(k<i)
    /// l_ik·y_k`, then `y_i - Σ_(k>i) l_ki·x_k`) followed by one correctly rounded division
    /// by the pivot: two roundings per intermediate, 8 divisions in total. Never a multiplication
    /// by a rounded reciprocal (see the module doc).
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Cholesky4<T>, b: Vector4<T>) -> Vector4<T> {
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
        let w = R::wide_add(R::wide_zero(), b.w);
        let w = R::wide_sub_prod(w, self.l41, y1);
        let w = R::wide_sub_prod(w, self.l42, y2);
        let w = R::wide_sub_prod(w, self.l43, y3);
        let f4 = R::wide_rescale(w);
        let y4 = R::div(f4, self.l44);
        let x4 = R::div(y4, self.l44);
        let w = R::wide_add(R::wide_zero(), y3);
        let w = R::wide_sub_prod(w, self.l43, x4);
        let g3 = R::wide_rescale(w);
        let x3 = R::div(g3, self.l33);
        let w = R::wide_add(R::wide_zero(), y2);
        let w = R::wide_sub_prod(w, self.l32, x3);
        let w = R::wide_sub_prod(w, self.l42, x4);
        let g2 = R::wide_rescale(w);
        let x2 = R::div(g2, self.l22);
        let w = R::wide_add(R::wide_zero(), y1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let w = R::wide_sub_prod(w, self.l31, x3);
        let w = R::wide_sub_prod(w, self.l41, x4);
        let g1 = R::wide_rescale(w);
        let x1 = R::div(g1, self.l11);
        Vector4 { x: x1, y: x2, z: x3, w: x4 }
    }

    /// `a⁻¹ = l⁻ᵀ · l⁻¹`, as a full (symmetric) `Matrix4`, like upstream.
    /// Upstream: `Cholesky::inverse`.
    ///
    /// `q = l⁻¹` (lower triangular) is built column by column — `q_jj = recip(l_jj)` (the
    /// `1 / l_jj` rounded to nearest, cheaper than `ONE / l_jj`) and
    /// `q_ij = (-Σ_(k=j..i-1) l_ik·q_kj) / l_ii`, one exact accumulation and one correctly
    /// rounded division each — then `a⁻¹_ij = Σ_k q_ki·q_kj` is one exact accumulation
    /// floored once per output.
    /// 10 divisions in total (the 3 quotients of row 4 through one `Real::div3`), against the 32
    /// of 4 `solve` calls on the columns of the identity.
    ///
    /// The result is symmetric by construction (one accumulation per unordered pair), so only its
    /// upper triangle is computed. Panics on overflow, which for an ill-conditioned `a` happens
    /// well before the mathematical inverse stops fitting in the scalar.
    fn inverse(self: Cholesky4<T>) -> Matrix4<T> {
        let q11 = R::recip(self.l11);
        let q22 = R::recip(self.l22);
        let q33 = R::recip(self.l33);
        let q44 = R::recip(self.l44);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l21, q11);
        let t21 = R::wide_rescale(w);
        let q21 = R::div(t21, self.l22);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l31, q11);
        let w = R::wide_sub_prod(w, self.l32, q21);
        let t31 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l32, q22);
        let t32 = R::wide_rescale(w);
        let q31 = R::div(t31, self.l33);
        let q32 = R::div(t32, self.l33);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l41, q11);
        let w = R::wide_sub_prod(w, self.l42, q21);
        let w = R::wide_sub_prod(w, self.l43, q31);
        let t41 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l42, q22);
        let w = R::wide_sub_prod(w, self.l43, q32);
        let t42 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l43, q33);
        let t43 = R::wide_rescale(w);
        let (q41, q42, q43) = R::div3(t41, t42, t43, self.l44);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q11, q11);
        let w = R::wide_add_prod(w, q21, q21);
        let w = R::wide_add_prod(w, q31, q31);
        let w = R::wide_add_prod(w, q41, q41);
        let r11 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q21, q22);
        let w = R::wide_add_prod(w, q31, q32);
        let w = R::wide_add_prod(w, q41, q42);
        let r12 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q31, q33);
        let w = R::wide_add_prod(w, q41, q43);
        let r13 = R::wide_rescale(w);
        let r14 = q41 * q44;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q22, q22);
        let w = R::wide_add_prod(w, q32, q32);
        let w = R::wide_add_prod(w, q42, q42);
        let r22 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q32, q33);
        let w = R::wide_add_prod(w, q42, q43);
        let r23 = R::wide_rescale(w);
        let r24 = q42 * q44;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q33, q33);
        let w = R::wide_add_prod(w, q43, q43);
        let r33 = R::wide_rescale(w);
        let r34 = q43 * q44;
        let r44 = R::sqr(q44);
        Matrix4 {
            m11: r11,
            m21: r12,
            m31: r13,
            m41: r14,
            m12: r12,
            m22: r22,
            m32: r23,
            m42: r24,
            m13: r13,
            m23: r23,
            m33: r33,
            m43: r34,
            m14: r14,
            m24: r24,
            m34: r34,
            m44: r44,
        }
    }

    /// `det(a) = Π l_jj²`, computed as `(Π l_jj)²`: a balanced product tree (3 floored
    /// products) then one `Real::sqr` (one more floor), i.e. 4 roundings instead of the
    /// 7 of `Π (l_jj²)`. Upstream: `Cholesky::determinant`.
    ///
    /// The intermediate `Π l_jj` is the square root of the result, so it overflows only if the
    /// determinant itself does — and then this panics. Underflow is silent (a determinant below
    /// 1 ulp floors to 0); that is the scalar's floor rounding, NOT a singularity report, since
    /// `new` already accepted the matrix.
    #[inline(always)]
    fn determinant(self: Cholesky4<T>) -> T {
        R::sqr((self.l11 * self.l22) * (self.l33 * self.l44))
    }
}

/// Methods of `Cholesky6<T>` for any `Real` scalar.
#[generate_trait]
pub impl Cholesky6Impl<
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
> of Cholesky6Trait<T> {
    /// The Cholesky factorisation `a = l * lᵀ` of the symmetric positive-definite `a`, or `None`
    /// when `a` is not positive definite. Upstream: `Cholesky::new`.
    ///
    /// Like upstream, only the LOWER triangle of `a` is read (the entries at row `i`,
    /// column `j` with `i >= j`): the strictly upper triangle is ignored and the symmetry
    /// of `a` is NOT checked.
    ///
    /// Column by column (j = 1..6): the pivot `p_j = a_jj - Σ_(k<j) l_jk²` is accumulated
    /// exactly in the wide accumulator and floored ONCE, then `l_jj = sqrt(p_j)` (exact floor of
    /// the square root); each sub-diagonal entry `l_ij = (a_ij - Σ_(k<j) l_ik·l_jk) / l_jj` costs
    /// one exact accumulation (one floor) and one correctly rounded division (a second rounding, to
    /// nearest).
    /// 6 square roots and 15 divisions in total.
    ///
    /// SINGULARITY CRITERION: `None` as soon as a pivot is `<= 0` AFTER that flooring. Upstream
    /// tests the real pivot against 0; flooring makes the test slightly stricter, so a matrix that
    /// is positive definite in exact arithmetic but whose j-th pivot is below 1 ulp (2^-32 in
    /// Q32.32) is rejected here — that pivot is indistinguishable from zero in the scalar and
    /// `sqrt` of it would carry no information. Indefinite or non-symmetric input is caught the
    /// same way, but only when it drives a pivot to zero: `new` is not a definiteness test.
    ///
    /// Panics on overflow of a pivot or a numerator; never wraps.
    fn new(a: Matrix6<T>) -> Option<Cholesky6<T>> {
        let p1 = a.m11.m11;
        if p1 <= R::zero() {
            return None;
        }
        let l11 = R::sqrt(p1);
        let (l21, l31, l41, l51, l61) = R::div5(
            a.m11.m21, a.m11.m31, a.m21.m11, a.m21.m21, a.m21.m31, l11,
        );
        let w = R::wide_add(R::wide_zero(), a.m11.m22);
        let w = R::wide_sub_prod(w, l21, l21);
        let p2 = R::wide_rescale(w);
        if p2 <= R::zero() {
            return None;
        }
        let l22 = R::sqrt(p2);
        let w = R::wide_add(R::wide_zero(), a.m11.m32);
        let w = R::wide_sub_prod(w, l31, l21);
        let n32 = R::wide_rescale(w);
        let l32 = R::div(n32, l22);
        let w = R::wide_add(R::wide_zero(), a.m21.m12);
        let w = R::wide_sub_prod(w, l41, l21);
        let n42 = R::wide_rescale(w);
        let l42 = R::div(n42, l22);
        let w = R::wide_add(R::wide_zero(), a.m21.m22);
        let w = R::wide_sub_prod(w, l51, l21);
        let n52 = R::wide_rescale(w);
        let l52 = R::div(n52, l22);
        let w = R::wide_add(R::wide_zero(), a.m21.m32);
        let w = R::wide_sub_prod(w, l61, l21);
        let n62 = R::wide_rescale(w);
        let l62 = R::div(n62, l22);
        let w = R::wide_add(R::wide_zero(), a.m11.m33);
        let w = R::wide_sub_prod(w, l31, l31);
        let w = R::wide_sub_prod(w, l32, l32);
        let p3 = R::wide_rescale(w);
        if p3 <= R::zero() {
            return None;
        }
        let l33 = R::sqrt(p3);
        let w = R::wide_add(R::wide_zero(), a.m21.m13);
        let w = R::wide_sub_prod(w, l41, l31);
        let w = R::wide_sub_prod(w, l42, l32);
        let n43 = R::wide_rescale(w);
        let l43 = R::div(n43, l33);
        let w = R::wide_add(R::wide_zero(), a.m21.m23);
        let w = R::wide_sub_prod(w, l51, l31);
        let w = R::wide_sub_prod(w, l52, l32);
        let n53 = R::wide_rescale(w);
        let l53 = R::div(n53, l33);
        let w = R::wide_add(R::wide_zero(), a.m21.m33);
        let w = R::wide_sub_prod(w, l61, l31);
        let w = R::wide_sub_prod(w, l62, l32);
        let n63 = R::wide_rescale(w);
        let l63 = R::div(n63, l33);
        let w = R::wide_add(R::wide_zero(), a.m22.m11);
        let w = R::wide_sub_prod(w, l41, l41);
        let w = R::wide_sub_prod(w, l42, l42);
        let w = R::wide_sub_prod(w, l43, l43);
        let p4 = R::wide_rescale(w);
        if p4 <= R::zero() {
            return None;
        }
        let l44 = R::sqrt(p4);
        let w = R::wide_add(R::wide_zero(), a.m22.m21);
        let w = R::wide_sub_prod(w, l51, l41);
        let w = R::wide_sub_prod(w, l52, l42);
        let w = R::wide_sub_prod(w, l53, l43);
        let n54 = R::wide_rescale(w);
        let l54 = R::div(n54, l44);
        let w = R::wide_add(R::wide_zero(), a.m22.m31);
        let w = R::wide_sub_prod(w, l61, l41);
        let w = R::wide_sub_prod(w, l62, l42);
        let w = R::wide_sub_prod(w, l63, l43);
        let n64 = R::wide_rescale(w);
        let l64 = R::div(n64, l44);
        let w = R::wide_add(R::wide_zero(), a.m22.m22);
        let w = R::wide_sub_prod(w, l51, l51);
        let w = R::wide_sub_prod(w, l52, l52);
        let w = R::wide_sub_prod(w, l53, l53);
        let w = R::wide_sub_prod(w, l54, l54);
        let p5 = R::wide_rescale(w);
        if p5 <= R::zero() {
            return None;
        }
        let l55 = R::sqrt(p5);
        let w = R::wide_add(R::wide_zero(), a.m22.m32);
        let w = R::wide_sub_prod(w, l61, l51);
        let w = R::wide_sub_prod(w, l62, l52);
        let w = R::wide_sub_prod(w, l63, l53);
        let w = R::wide_sub_prod(w, l64, l54);
        let n65 = R::wide_rescale(w);
        let l65 = R::div(n65, l55);
        let w = R::wide_add(R::wide_zero(), a.m22.m33);
        let w = R::wide_sub_prod(w, l61, l61);
        let w = R::wide_sub_prod(w, l62, l62);
        let w = R::wide_sub_prod(w, l63, l63);
        let w = R::wide_sub_prod(w, l64, l64);
        let w = R::wide_sub_prod(w, l65, l65);
        let p6 = R::wide_rescale(w);
        if p6 <= R::zero() {
            return None;
        }
        let l66 = R::sqrt(p6);
        Some(
            Cholesky6 {
                l11,
                l21,
                l31,
                l41,
                l51,
                l61,
                l22,
                l32,
                l42,
                l52,
                l62,
                l33,
                l43,
                l53,
                l63,
                l44,
                l54,
                l64,
                l55,
                l65,
                l66,
            },
        )
    }

    /// The lower triangular factor, with explicit zeros above the diagonal. Upstream:
    /// `Cholesky::l`. Exact: the stored components are copied, nothing is recomputed.
    #[inline(always)]
    fn l(self: Cholesky6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3 {
                m11: self.l11,
                m21: self.l21,
                m31: self.l31,
                m12: R::zero(),
                m22: self.l22,
                m32: self.l32,
                m13: R::zero(),
                m23: R::zero(),
                m33: self.l33,
            },
            m21: Matrix3 {
                m11: self.l41,
                m21: self.l51,
                m31: self.l61,
                m12: self.l42,
                m22: self.l52,
                m32: self.l62,
                m13: self.l43,
                m23: self.l53,
                m33: self.l63,
            },
            m12: Matrix3 {
                m11: R::zero(),
                m21: R::zero(),
                m31: R::zero(),
                m12: R::zero(),
                m22: R::zero(),
                m32: R::zero(),
                m13: R::zero(),
                m23: R::zero(),
                m33: R::zero(),
            },
            m22: Matrix3 {
                m11: self.l44,
                m21: self.l54,
                m31: self.l64,
                m12: R::zero(),
                m22: self.l55,
                m32: self.l65,
                m13: R::zero(),
                m23: R::zero(),
                m33: self.l66,
            },
        }
    }

    /// The solution `x` of `a * x = b`, by forward substitution on `l` then back substitution on
    /// `lᵀ`. Upstream: `Cholesky::solve`.
    ///
    /// Each of the 12 intermediates is ONE exact accumulation floored once (`b_i - Σ_(k<i)
    /// l_ik·y_k`, then `y_i - Σ_(k>i) l_ki·x_k`) followed by one correctly rounded division
    /// by the pivot: two roundings per intermediate, 12 divisions in total. Never a multiplication
    /// by a rounded reciprocal (see the module doc).
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Cholesky6<T>, b: Vector6<T>) -> Vector6<T> {
        let y1 = R::div(b.a.x, self.l11);
        let w = R::wide_add(R::wide_zero(), b.a.y);
        let w = R::wide_sub_prod(w, self.l21, y1);
        let f2 = R::wide_rescale(w);
        let y2 = R::div(f2, self.l22);
        let w = R::wide_add(R::wide_zero(), b.a.z);
        let w = R::wide_sub_prod(w, self.l31, y1);
        let w = R::wide_sub_prod(w, self.l32, y2);
        let f3 = R::wide_rescale(w);
        let y3 = R::div(f3, self.l33);
        let w = R::wide_add(R::wide_zero(), b.b.x);
        let w = R::wide_sub_prod(w, self.l41, y1);
        let w = R::wide_sub_prod(w, self.l42, y2);
        let w = R::wide_sub_prod(w, self.l43, y3);
        let f4 = R::wide_rescale(w);
        let y4 = R::div(f4, self.l44);
        let w = R::wide_add(R::wide_zero(), b.b.y);
        let w = R::wide_sub_prod(w, self.l51, y1);
        let w = R::wide_sub_prod(w, self.l52, y2);
        let w = R::wide_sub_prod(w, self.l53, y3);
        let w = R::wide_sub_prod(w, self.l54, y4);
        let f5 = R::wide_rescale(w);
        let y5 = R::div(f5, self.l55);
        let w = R::wide_add(R::wide_zero(), b.b.z);
        let w = R::wide_sub_prod(w, self.l61, y1);
        let w = R::wide_sub_prod(w, self.l62, y2);
        let w = R::wide_sub_prod(w, self.l63, y3);
        let w = R::wide_sub_prod(w, self.l64, y4);
        let w = R::wide_sub_prod(w, self.l65, y5);
        let f6 = R::wide_rescale(w);
        let y6 = R::div(f6, self.l66);
        let x6 = R::div(y6, self.l66);
        let w = R::wide_add(R::wide_zero(), y5);
        let w = R::wide_sub_prod(w, self.l65, x6);
        let g5 = R::wide_rescale(w);
        let x5 = R::div(g5, self.l55);
        let w = R::wide_add(R::wide_zero(), y4);
        let w = R::wide_sub_prod(w, self.l54, x5);
        let w = R::wide_sub_prod(w, self.l64, x6);
        let g4 = R::wide_rescale(w);
        let x4 = R::div(g4, self.l44);
        let w = R::wide_add(R::wide_zero(), y3);
        let w = R::wide_sub_prod(w, self.l43, x4);
        let w = R::wide_sub_prod(w, self.l53, x5);
        let w = R::wide_sub_prod(w, self.l63, x6);
        let g3 = R::wide_rescale(w);
        let x3 = R::div(g3, self.l33);
        let w = R::wide_add(R::wide_zero(), y2);
        let w = R::wide_sub_prod(w, self.l32, x3);
        let w = R::wide_sub_prod(w, self.l42, x4);
        let w = R::wide_sub_prod(w, self.l52, x5);
        let w = R::wide_sub_prod(w, self.l62, x6);
        let g2 = R::wide_rescale(w);
        let x2 = R::div(g2, self.l22);
        let w = R::wide_add(R::wide_zero(), y1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let w = R::wide_sub_prod(w, self.l31, x3);
        let w = R::wide_sub_prod(w, self.l41, x4);
        let w = R::wide_sub_prod(w, self.l51, x5);
        let w = R::wide_sub_prod(w, self.l61, x6);
        let g1 = R::wide_rescale(w);
        let x1 = R::div(g1, self.l11);
        Vector6 { a: Vector3 { x: x1, y: x2, z: x3 }, b: Vector3 { x: x4, y: x5, z: x6 } }
    }

    /// `a⁻¹ = l⁻ᵀ · l⁻¹`, as a full (symmetric) `Matrix6`, like upstream.
    /// Upstream: `Cholesky::inverse`.
    ///
    /// `q = l⁻¹` (lower triangular) is built column by column — `q_jj = recip(l_jj)` (the
    /// `1 / l_jj` rounded to nearest, cheaper than `ONE / l_jj`) and
    /// `q_ij = (-Σ_(k=j..i-1) l_ik·q_kj) / l_ii`, one exact accumulation and one correctly
    /// rounded division each — then `a⁻¹_ij = Σ_k q_ki·q_kj` is one exact accumulation
    /// floored once per output.
    /// 21 divisions in total (rows 4, 5 and 6 through one `Real::div3` / `div4` / `div5` each),
    /// against the 72 of 6 `solve` calls on the columns of the identity.
    ///
    /// The result is symmetric by construction (one accumulation per unordered pair), so only its
    /// upper triangle is computed. Panics on overflow, which for an ill-conditioned `a` happens
    /// well before the mathematical inverse stops fitting in the scalar.
    fn inverse(self: Cholesky6<T>) -> Matrix6<T> {
        let q11 = R::recip(self.l11);
        let q22 = R::recip(self.l22);
        let q33 = R::recip(self.l33);
        let q44 = R::recip(self.l44);
        let q55 = R::recip(self.l55);
        let q66 = R::recip(self.l66);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l21, q11);
        let t21 = R::wide_rescale(w);
        let q21 = R::div(t21, self.l22);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l31, q11);
        let w = R::wide_sub_prod(w, self.l32, q21);
        let t31 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l32, q22);
        let t32 = R::wide_rescale(w);
        let q31 = R::div(t31, self.l33);
        let q32 = R::div(t32, self.l33);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l41, q11);
        let w = R::wide_sub_prod(w, self.l42, q21);
        let w = R::wide_sub_prod(w, self.l43, q31);
        let t41 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l42, q22);
        let w = R::wide_sub_prod(w, self.l43, q32);
        let t42 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l43, q33);
        let t43 = R::wide_rescale(w);
        let (q41, q42, q43) = R::div3(t41, t42, t43, self.l44);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l51, q11);
        let w = R::wide_sub_prod(w, self.l52, q21);
        let w = R::wide_sub_prod(w, self.l53, q31);
        let w = R::wide_sub_prod(w, self.l54, q41);
        let t51 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l52, q22);
        let w = R::wide_sub_prod(w, self.l53, q32);
        let w = R::wide_sub_prod(w, self.l54, q42);
        let t52 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l53, q33);
        let w = R::wide_sub_prod(w, self.l54, q43);
        let t53 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l54, q44);
        let t54 = R::wide_rescale(w);
        let (q51, q52, q53, q54) = R::div4(t51, t52, t53, t54, self.l55);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l61, q11);
        let w = R::wide_sub_prod(w, self.l62, q21);
        let w = R::wide_sub_prod(w, self.l63, q31);
        let w = R::wide_sub_prod(w, self.l64, q41);
        let w = R::wide_sub_prod(w, self.l65, q51);
        let t61 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l62, q22);
        let w = R::wide_sub_prod(w, self.l63, q32);
        let w = R::wide_sub_prod(w, self.l64, q42);
        let w = R::wide_sub_prod(w, self.l65, q52);
        let t62 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l63, q33);
        let w = R::wide_sub_prod(w, self.l64, q43);
        let w = R::wide_sub_prod(w, self.l65, q53);
        let t63 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l64, q44);
        let w = R::wide_sub_prod(w, self.l65, q54);
        let t64 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_sub_prod(w, self.l65, q55);
        let t65 = R::wide_rescale(w);
        let (q61, q62, q63, q64, q65) = R::div5(t61, t62, t63, t64, t65, self.l66);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q11, q11);
        let w = R::wide_add_prod(w, q21, q21);
        let w = R::wide_add_prod(w, q31, q31);
        let w = R::wide_add_prod(w, q41, q41);
        let w = R::wide_add_prod(w, q51, q51);
        let w = R::wide_add_prod(w, q61, q61);
        let r11 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q21, q22);
        let w = R::wide_add_prod(w, q31, q32);
        let w = R::wide_add_prod(w, q41, q42);
        let w = R::wide_add_prod(w, q51, q52);
        let w = R::wide_add_prod(w, q61, q62);
        let r12 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q31, q33);
        let w = R::wide_add_prod(w, q41, q43);
        let w = R::wide_add_prod(w, q51, q53);
        let w = R::wide_add_prod(w, q61, q63);
        let r13 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q41, q44);
        let w = R::wide_add_prod(w, q51, q54);
        let w = R::wide_add_prod(w, q61, q64);
        let r14 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q51, q55);
        let w = R::wide_add_prod(w, q61, q65);
        let r15 = R::wide_rescale(w);
        let r16 = q61 * q66;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q22, q22);
        let w = R::wide_add_prod(w, q32, q32);
        let w = R::wide_add_prod(w, q42, q42);
        let w = R::wide_add_prod(w, q52, q52);
        let w = R::wide_add_prod(w, q62, q62);
        let r22 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q32, q33);
        let w = R::wide_add_prod(w, q42, q43);
        let w = R::wide_add_prod(w, q52, q53);
        let w = R::wide_add_prod(w, q62, q63);
        let r23 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q42, q44);
        let w = R::wide_add_prod(w, q52, q54);
        let w = R::wide_add_prod(w, q62, q64);
        let r24 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q52, q55);
        let w = R::wide_add_prod(w, q62, q65);
        let r25 = R::wide_rescale(w);
        let r26 = q62 * q66;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q33, q33);
        let w = R::wide_add_prod(w, q43, q43);
        let w = R::wide_add_prod(w, q53, q53);
        let w = R::wide_add_prod(w, q63, q63);
        let r33 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q43, q44);
        let w = R::wide_add_prod(w, q53, q54);
        let w = R::wide_add_prod(w, q63, q64);
        let r34 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q53, q55);
        let w = R::wide_add_prod(w, q63, q65);
        let r35 = R::wide_rescale(w);
        let r36 = q63 * q66;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q44, q44);
        let w = R::wide_add_prod(w, q54, q54);
        let w = R::wide_add_prod(w, q64, q64);
        let r44 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q54, q55);
        let w = R::wide_add_prod(w, q64, q65);
        let r45 = R::wide_rescale(w);
        let r46 = q64 * q66;
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q55, q55);
        let w = R::wide_add_prod(w, q65, q65);
        let r55 = R::wide_rescale(w);
        let r56 = q65 * q66;
        let r66 = R::sqr(q66);
        Matrix6 {
            m11: Matrix3 {
                m11: r11,
                m21: r12,
                m31: r13,
                m12: r12,
                m22: r22,
                m32: r23,
                m13: r13,
                m23: r23,
                m33: r33,
            },
            m21: Matrix3 {
                m11: r14,
                m21: r15,
                m31: r16,
                m12: r24,
                m22: r25,
                m32: r26,
                m13: r34,
                m23: r35,
                m33: r36,
            },
            m12: Matrix3 {
                m11: r14,
                m21: r24,
                m31: r34,
                m12: r15,
                m22: r25,
                m32: r35,
                m13: r16,
                m23: r26,
                m33: r36,
            },
            m22: Matrix3 {
                m11: r44,
                m21: r45,
                m31: r46,
                m12: r45,
                m22: r55,
                m32: r56,
                m13: r46,
                m23: r56,
                m33: r66,
            },
        }
    }

    /// `det(a) = Π l_jj²`, computed as `(Π l_jj)²`: a balanced product tree (5 floored
    /// products) then one `Real::sqr` (one more floor), i.e. 6 roundings instead of the
    /// 11 of `Π (l_jj²)`. Upstream: `Cholesky::determinant`.
    ///
    /// The intermediate `Π l_jj` is the square root of the result, so it overflows only if the
    /// determinant itself does — and then this panics. Underflow is silent (a determinant below
    /// 1 ulp floors to 0); that is the scalar's floor rounding, NOT a singularity report, since
    /// `new` already accepted the matrix.
    #[inline(always)]
    fn determinant(self: Cholesky6<T>) -> T {
        R::sqr((self.l11 * (self.l22 * self.l33)) * (self.l44 * (self.l55 * self.l66)))
    }
}
