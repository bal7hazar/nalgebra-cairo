//! `LDLᵀ` factorisation `A = L·D·Lᵀ` of a symmetric matrix — `L` unit lower triangular, `D`
//! diagonal — unrolled for the static sizes 2, 3, 4 and 6. No square root anywhere, which is why
//! DESIGN D6 prefers it to `crate::linalg::cholesky` whenever a factor is only a means to solve a
//! system.
//!
//! UPSTREAM MAPPING. `nalgebra` ships the mirror image, `nalgebra::linalg::UDU`:
//! `A = U·D'·Uᵀ` with `U` unit UPPER triangular, which factors the trailing submatrices instead
//! of the leading ones. With `J` the reversal permutation (`J_ij = 1` iff `i + j = n + 1`), the two
//! are the same algorithm read backwards:
//!
//! ```text
//! U = J·L'·J        D' = J·D''·J        where (L', D'') = LDLᵀ(J·A·J)
//! ```
//!
//! so `UDU::new(a)` and `Ldlt::new(reverse(a))` carry the same information, and the systems they
//! solve are identical: `udu{n}_solve` / `udu{n}_inverse` oracle vectors are valid for both, and
//! the `ldlt{n}_l_d` vectors give the factors in THIS convention. `tests.cairo` checks the
//! mapping explicitly for n = 2 and 3 against the `udu{n}_u_d` vectors.
//!
//! LDLᵀ is chosen over UDU because `L` unit LOWER is the convention of every other
//! lower-triangular factor here (`Cholesky::l`) and of rapier's solvers, and because a
//! factorisation that consumes the leading principal minors in order matches the block layout of
//! `Matrix6`.
//!
//! What is stored: the n(n-1)/2 strictly lower components of `L` as flat named fields (the unit
//! diagonal is implicit and never materialised) plus `D` as a `VectorN`, returned as such by `d()`.
//!
//! Numeric contract (AGENTS.md rule 4): every sum of products is accumulated EXACTLY in the
//! `Real::Wide` accumulator and floored ONCE; divisions are truncated divisions, never a
//! multiplication by a rounded reciprocal (the `alt_recip` candidates of `benches.cairo` lose on
//! gas in `solve` and on accuracy in `inverse`). `new` keeps the unrounded numerator of each
//! column as the column of `l·diag(d)` instead of recomputing `l_jk·d_k`: the `alt_products`
//! candidate that recomputes it is measurably dearer and less accurate, see `Ldlt2Trait::new`.
//!
//! NOT ported: upstream's `UDU` has no `solve` / `inverse` / `determinant` at all (it only exposes
//! `u` and `d`); those follow `Cholesky`'s surface here. `rank_one_update` is not ported either,
//! for the reason given in `cholesky.cairo`.

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

/// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of a symmetric 2x2 matrix: the 1
/// strictly lower components of the UNIT lower triangular `l` (its diagonal is an implicit 1 and
/// is not stored) and the diagonal `d`.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I > J`). Fields are declared column by
/// column, then `d`, so `Serde` writes `l21, .., l21, l32, .., d`. Build one with
/// `Ldlt2Trait::new`; the fields are public so that a factor computed elsewhere can be
/// re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Ldlt2<T> {
    /// Row 2, column 1 of the unit lower triangular factor.
    pub l21: T,
    /// The diagonal of `diag(d)`.
    pub d: Vector2<T>,
}

/// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of a symmetric 3x3 matrix: the 3
/// strictly lower components of the UNIT lower triangular `l` (its diagonal is an implicit 1 and
/// is not stored) and the diagonal `d`.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I > J`). Fields are declared column by
/// column, then `d`, so `Serde` writes `l21, .., l31, l32, .., d`. Build one with
/// `Ldlt3Trait::new`; the fields are public so that a factor computed elsewhere can be
/// re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Ldlt3<T> {
    /// Row 2, column 1 of the unit lower triangular factor.
    pub l21: T,
    /// Row 3, column 1 of the unit lower triangular factor.
    pub l31: T,
    /// Row 3, column 2 of the unit lower triangular factor.
    pub l32: T,
    /// The diagonal of `diag(d)`.
    pub d: Vector3<T>,
}

/// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of a symmetric 4x4 matrix: the 6
/// strictly lower components of the UNIT lower triangular `l` (its diagonal is an implicit 1 and
/// is not stored) and the diagonal `d`.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I > J`). Fields are declared column by
/// column, then `d`, so `Serde` writes `l21, .., l41, l32, .., d`. Build one with
/// `Ldlt4Trait::new`; the fields are public so that a factor computed elsewhere can be
/// re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Ldlt4<T> {
    /// Row 2, column 1 of the unit lower triangular factor.
    pub l21: T,
    /// Row 3, column 1 of the unit lower triangular factor.
    pub l31: T,
    /// Row 4, column 1 of the unit lower triangular factor.
    pub l41: T,
    /// Row 3, column 2 of the unit lower triangular factor.
    pub l32: T,
    /// Row 4, column 2 of the unit lower triangular factor.
    pub l42: T,
    /// Row 4, column 3 of the unit lower triangular factor.
    pub l43: T,
    /// The diagonal of `diag(d)`.
    pub d: Vector4<T>,
}

/// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of a symmetric 6x6 matrix: the 15
/// strictly lower components of the UNIT lower triangular `l` (its diagonal is an implicit 1 and
/// is not stored) and the diagonal `d`.
///
/// `lIJ` is the entry at row `I`, column `J` (1-based, `I > J`). Fields are declared column by
/// column, then `d`, so `Serde` writes `l21, .., l61, l32, .., d`. Build one with
/// `Ldlt6Trait::new`; the fields are public so that a factor computed elsewhere can be
/// re-assembled, and nothing checks that they form a valid factor.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Ldlt6<T> {
    /// Row 2, column 1 of the unit lower triangular factor.
    pub l21: T,
    /// Row 3, column 1 of the unit lower triangular factor.
    pub l31: T,
    /// Row 4, column 1 of the unit lower triangular factor.
    pub l41: T,
    /// Row 5, column 1 of the unit lower triangular factor.
    pub l51: T,
    /// Row 6, column 1 of the unit lower triangular factor.
    pub l61: T,
    /// Row 3, column 2 of the unit lower triangular factor.
    pub l32: T,
    /// Row 4, column 2 of the unit lower triangular factor.
    pub l42: T,
    /// Row 5, column 2 of the unit lower triangular factor.
    pub l52: T,
    /// Row 6, column 2 of the unit lower triangular factor.
    pub l62: T,
    /// Row 4, column 3 of the unit lower triangular factor.
    pub l43: T,
    /// Row 5, column 3 of the unit lower triangular factor.
    pub l53: T,
    /// Row 6, column 3 of the unit lower triangular factor.
    pub l63: T,
    /// Row 5, column 4 of the unit lower triangular factor.
    pub l54: T,
    /// Row 6, column 4 of the unit lower triangular factor.
    pub l64: T,
    /// Row 6, column 5 of the unit lower triangular factor.
    pub l65: T,
    /// The diagonal of `diag(d)`.
    pub d: Vector6<T>,
}

/// Methods of `Ldlt2<T>` for any `Real` scalar.
#[generate_trait]
pub impl Ldlt2Impl<
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
> of Ldlt2Trait<T> {
    /// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of the symmetric `a`, or `None` when a
    /// pivot is exactly zero. Upstream: `UDU::new` on the reversed matrix (see the module doc).
    ///
    /// `a` is a `SymMatrix2`, so there is no triangle to choose: its 3 independent
    /// components ARE the matrix.
    ///
    /// Column by column (j = 1..2), in terms of the numerators `n_ij = a_ij - Σ_(k<j) l_ik·n_jk`
    /// for `i >= j`: each is ONE exact accumulation floored ONCE, the pivot is `d_j = n_jj`, and
    /// `l_ij = n_ij / d_j` is one truncated division. 1 division, no square root
    /// and no other product.
    ///
    /// `n_jk` IS the column of `l·diag(d)`: `l_jk·d_k` differs from it only by the remainder of
    /// the division that produced `l_jk`. Recomputing `l_jk·d_k` explicitly (the `alt_products`
    /// candidate of `benches.cairo`) costs 1 extra product, makes `new` 30 to 32 %
    /// dearer, and moves the factors FARTHER from the true ones: `a_jj - l_jk·(l_jk·d_k)` is
    /// second order in the rounding error of `l_jk` where `a_jj - l_jk·n_jk` is only first order
    /// (16 ulp against 34 at size 2, 27 against 42 at size 4). The backward error
    /// `l·diag(d)·lᵀ - a` is identical for both, being dominated by `l_ij·d_j - a_ij`, which
    /// no choice here changes — so the cheaper and more accurate form is the one that ships.
    ///
    /// SINGULARITY CRITERION: `None` iff a pivot `d_j` is EXACTLY zero, i.e. iff the leading
    /// principal minor of order j is zero after flooring. Unlike `Cholesky::new`, a negative
    /// pivot is accepted: `LDLᵀ` factorises any symmetric matrix whose leading principal minors
    /// are non-zero, indefinite ones included, and `d` then has mixed signs. It is NOT a
    /// definiteness test, and it has no pivoting: a symmetric matrix with a zero leading minor
    /// (`[[0, 1], [1, 0]]`) is rejected although it is invertible.
    ///
    /// Panics on overflow of a pivot or a numerator; never wraps.
    fn new(a: SymMatrix2<T>) -> Option<Ldlt2<T>> {
        let d1 = a.m11;
        if d1 == R::ZERO {
            return None;
        }
        let n21 = a.m12;
        let l21 = R::div(n21, d1);
        let w = R::wide_add(R::wide_zero(), a.m22);
        let w = R::wide_sub_prod(w, l21, n21);
        let d2 = R::wide_rescale(w);
        if d2 == R::ZERO {
            return None;
        }
        Some(Ldlt2 { l21, d: Vector2 { x: d1, y: d2 } })
    }

    /// The unit lower triangular factor, with explicit ones on the diagonal and zeros above it.
    /// Upstream: the reverse-permuted `UDU::u`. Exact: nothing is recomputed.
    #[inline(always)]
    fn l(self: Ldlt2<T>) -> Matrix2<T> {
        Matrix2 { m11: R::ONE, m21: self.l21, m12: R::ZERO, m22: R::ONE }
    }

    /// The diagonal of `diag(d)`. Upstream: `UDU::d`, reversed. Exact: the stored vector.
    #[inline(always)]
    fn d(self: Ldlt2<T>) -> Vector2<T> {
        self.d
    }

    /// The solution `x` of `a * x = b`: forward substitution on the unit `l`, division by `d`,
    /// back substitution on the unit `lᵀ`. Upstream: `UDU` has no `solve`; the oracle's
    /// `udu2_solve` vectors apply unchanged.
    ///
    /// The unit diagonal removes the divisions of the two substitutions, so this costs 2
    /// divisions where `Cholesky2::solve` costs 4: each of the 4 substitution
    /// intermediates is ONE exact accumulation floored once and nothing more, and only the
    /// diagonal solve `z_i = y_i / d_i` divides.
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Ldlt2<T>, b: Vector2<T>) -> Vector2<T> {
        let y1 = b.x;
        let w = R::wide_add(R::wide_zero(), b.y);
        let w = R::wide_sub_prod(w, self.l21, y1);
        let y2 = R::wide_rescale(w);
        let z1 = R::div(y1, self.d.x);
        let z2 = R::div(y2, self.d.y);
        let x2 = z2;
        let w = R::wide_add(R::wide_zero(), z1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let x1 = R::wide_rescale(w);
        Vector2 { x: x1, y: x2 }
    }

    /// `a⁻¹ = l⁻ᵀ · diag(d)⁻¹ · l⁻¹`, as its 3 independent components.
    ///
    /// `q = l⁻¹` is unit lower triangular and is built WITHOUT any division
    /// (`q_ij = -(l_ij + Σ_(k=j+1..i-1) l_ik·q_kj)`, one exact accumulation floored once); the
    /// 3 scaled entries `s_kj = q_kj / d_k` are the only divisions (2 of them are
    /// `recip(d_k)`, the `1 / d_k` truncated toward zero), and `a⁻¹_ij = Σ_k q_ki·s_kj` is one
    /// exact accumulation floored once per output component.
    ///
    /// The result is symmetric by construction, so only its upper triangle is computed — the sum
    /// is taken as `Σ_k q_ki·s_kj` with `i <= j`, which for rounded `s` differs from
    /// `Σ_k q_kj·s_ki` by at most a few ulp. Panics on overflow.
    fn inverse(self: Ldlt2<T>) -> SymMatrix2<T> {
        let q21 = -self.l21;
        let s11 = R::recip(self.d.x);
        let s21 = R::div(q21, self.d.y);
        let s22 = R::recip(self.d.y);
        let w = R::wide_zero();
        let w = R::wide_add(w, s11);
        let w = R::wide_add_prod(w, q21, s21);
        let r11 = R::wide_rescale(w);
        let r12 = q21 * s22;
        let r22 = s22;
        SymMatrix2 { m11: r11, m12: r12, m22: r22 }
    }

    /// `det(a) = Π d_j`, as a balanced product tree: 1 floored product, the
    /// shallowest chain of roundings for 2 factors. `det(l) = 1`, so no other term appears.
    ///
    /// Panics if the determinant does not fit the scalar. Underflow is silent (a determinant below
    /// 1 ulp floors to 0), which is the scalar's floor rounding and NOT a singularity report.
    #[inline(always)]
    fn determinant(self: Ldlt2<T>) -> T {
        self.d.x * self.d.y
    }
}

/// Methods of `Ldlt3<T>` for any `Real` scalar.
#[generate_trait]
pub impl Ldlt3Impl<
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
> of Ldlt3Trait<T> {
    /// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of the symmetric `a`, or `None` when a
    /// pivot is exactly zero. Upstream: `UDU::new` on the reversed matrix (see the module doc).
    ///
    /// `a` is a `SymMatrix3`, so there is no triangle to choose: its 6 independent
    /// components ARE the matrix.
    ///
    /// Column by column (j = 1..3), in terms of the numerators `n_ij = a_ij - Σ_(k<j) l_ik·n_jk`
    /// for `i >= j`: each is ONE exact accumulation floored ONCE, the pivot is `d_j = n_jj`, and
    /// `l_ij = n_ij / d_j` is one truncated division. 3 divisions, no square root
    /// and no other product.
    ///
    /// `n_jk` IS the column of `l·diag(d)`: `l_jk·d_k` differs from it only by the remainder of
    /// the division that produced `l_jk`. Recomputing `l_jk·d_k` explicitly (the `alt_products`
    /// candidate of `benches.cairo`) costs 3 extra products, makes `new` 30 to 32 %
    /// dearer, and moves the factors FARTHER from the true ones: `a_jj - l_jk·(l_jk·d_k)` is
    /// second order in the rounding error of `l_jk` where `a_jj - l_jk·n_jk` is only first order
    /// (16 ulp against 34 at size 2, 27 against 42 at size 4). The backward error
    /// `l·diag(d)·lᵀ - a` is identical for both, being dominated by `l_ij·d_j - a_ij`, which
    /// no choice here changes — so the cheaper and more accurate form is the one that ships.
    ///
    /// SINGULARITY CRITERION: `None` iff a pivot `d_j` is EXACTLY zero, i.e. iff the leading
    /// principal minor of order j is zero after flooring. Unlike `Cholesky::new`, a negative
    /// pivot is accepted: `LDLᵀ` factorises any symmetric matrix whose leading principal minors
    /// are non-zero, indefinite ones included, and `d` then has mixed signs. It is NOT a
    /// definiteness test, and it has no pivoting: a symmetric matrix with a zero leading minor
    /// (`[[0, 1], [1, 0]]`) is rejected although it is invertible.
    ///
    /// Panics on overflow of a pivot or a numerator; never wraps.
    fn new(a: SymMatrix3<T>) -> Option<Ldlt3<T>> {
        let d1 = a.m11;
        if d1 == R::ZERO {
            return None;
        }
        let n21 = a.m12;
        let l21 = R::div(n21, d1);
        let n31 = a.m13;
        let l31 = R::div(n31, d1);
        let w = R::wide_add(R::wide_zero(), a.m22);
        let w = R::wide_sub_prod(w, l21, n21);
        let d2 = R::wide_rescale(w);
        if d2 == R::ZERO {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m23);
        let w = R::wide_sub_prod(w, l31, n21);
        let n32 = R::wide_rescale(w);
        let l32 = R::div(n32, d2);
        let w = R::wide_add(R::wide_zero(), a.m33);
        let w = R::wide_sub_prod(w, l31, n31);
        let w = R::wide_sub_prod(w, l32, n32);
        let d3 = R::wide_rescale(w);
        if d3 == R::ZERO {
            return None;
        }
        Some(Ldlt3 { l21, l31, l32, d: Vector3 { x: d1, y: d2, z: d3 } })
    }

    /// The unit lower triangular factor, with explicit ones on the diagonal and zeros above it.
    /// Upstream: the reverse-permuted `UDU::u`. Exact: nothing is recomputed.
    #[inline(always)]
    fn l(self: Ldlt3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: R::ONE,
            m21: self.l21,
            m31: self.l31,
            m12: R::ZERO,
            m22: R::ONE,
            m32: self.l32,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: R::ONE,
        }
    }

    /// The diagonal of `diag(d)`. Upstream: `UDU::d`, reversed. Exact: the stored vector.
    #[inline(always)]
    fn d(self: Ldlt3<T>) -> Vector3<T> {
        self.d
    }

    /// The solution `x` of `a * x = b`: forward substitution on the unit `l`, division by `d`,
    /// back substitution on the unit `lᵀ`. Upstream: `UDU` has no `solve`; the oracle's
    /// `udu3_solve` vectors apply unchanged.
    ///
    /// The unit diagonal removes the divisions of the two substitutions, so this costs 3
    /// divisions where `Cholesky3::solve` costs 6: each of the 6 substitution
    /// intermediates is ONE exact accumulation floored once and nothing more, and only the
    /// diagonal solve `z_i = y_i / d_i` divides.
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Ldlt3<T>, b: Vector3<T>) -> Vector3<T> {
        let y1 = b.x;
        let w = R::wide_add(R::wide_zero(), b.y);
        let w = R::wide_sub_prod(w, self.l21, y1);
        let y2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.z);
        let w = R::wide_sub_prod(w, self.l31, y1);
        let w = R::wide_sub_prod(w, self.l32, y2);
        let y3 = R::wide_rescale(w);
        let z1 = R::div(y1, self.d.x);
        let z2 = R::div(y2, self.d.y);
        let z3 = R::div(y3, self.d.z);
        let x3 = z3;
        let w = R::wide_add(R::wide_zero(), z2);
        let w = R::wide_sub_prod(w, self.l32, x3);
        let x2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let w = R::wide_sub_prod(w, self.l31, x3);
        let x1 = R::wide_rescale(w);
        Vector3 { x: x1, y: x2, z: x3 }
    }

    /// `a⁻¹ = l⁻ᵀ · diag(d)⁻¹ · l⁻¹`, as its 6 independent components.
    ///
    /// `q = l⁻¹` is unit lower triangular and is built WITHOUT any division
    /// (`q_ij = -(l_ij + Σ_(k=j+1..i-1) l_ik·q_kj)`, one exact accumulation floored once); the
    /// 6 scaled entries `s_kj = q_kj / d_k` are the only divisions (3 of them are
    /// `recip(d_k)`, the `1 / d_k` truncated toward zero), and `a⁻¹_ij = Σ_k q_ki·s_kj` is one
    /// exact accumulation floored once per output component.
    ///
    /// The result is symmetric by construction, so only its upper triangle is computed — the sum
    /// is taken as `Σ_k q_ki·s_kj` with `i <= j`, which for rounded `s` differs from
    /// `Σ_k q_kj·s_ki` by at most a few ulp. Panics on overflow.
    fn inverse(self: Ldlt3<T>) -> SymMatrix3<T> {
        let q21 = -self.l21;
        let w = R::wide_sub(R::wide_zero(), self.l31);
        let w = R::wide_sub_prod(w, self.l32, q21);
        let q31 = R::wide_rescale(w);
        let q32 = -self.l32;
        let s11 = R::recip(self.d.x);
        let s21 = R::div(q21, self.d.y);
        let s31 = R::div(q31, self.d.z);
        let s22 = R::recip(self.d.y);
        let s32 = R::div(q32, self.d.z);
        let s33 = R::recip(self.d.z);
        let w = R::wide_zero();
        let w = R::wide_add(w, s11);
        let w = R::wide_add_prod(w, q21, s21);
        let w = R::wide_add_prod(w, q31, s31);
        let r11 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q21, s22);
        let w = R::wide_add_prod(w, q31, s32);
        let r12 = R::wide_rescale(w);
        let r13 = q31 * s33;
        let w = R::wide_zero();
        let w = R::wide_add(w, s22);
        let w = R::wide_add_prod(w, q32, s32);
        let r22 = R::wide_rescale(w);
        let r23 = q32 * s33;
        let r33 = s33;
        SymMatrix3 { m11: r11, m12: r12, m13: r13, m22: r22, m23: r23, m33: r33 }
    }

    /// `det(a) = Π d_j`, as a balanced product tree: 2 floored products, the
    /// shallowest chain of roundings for 3 factors. `det(l) = 1`, so no other term appears.
    ///
    /// Panics if the determinant does not fit the scalar. Underflow is silent (a determinant below
    /// 1 ulp floors to 0), which is the scalar's floor rounding and NOT a singularity report.
    #[inline(always)]
    fn determinant(self: Ldlt3<T>) -> T {
        self.d.x * (self.d.y * self.d.z)
    }
}

/// Methods of `Ldlt4<T>` for any `Real` scalar.
#[generate_trait]
pub impl Ldlt4Impl<
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
> of Ldlt4Trait<T> {
    /// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of the symmetric `a`, or `None` when a
    /// pivot is exactly zero. Upstream: `UDU::new` on the reversed matrix (see the module doc).
    ///
    /// Like upstream, only the LOWER triangle of `a` is read (the entries at row `i`,
    /// column `j` with `i >= j`): the strictly upper triangle is ignored and the symmetry
    /// of `a` is NOT checked. There is no `SymMatrix4` type, so a symmetric 4x4
    /// matrix travels as a plain `Matrix4`.
    ///
    /// Column by column (j = 1..4), in terms of the numerators `n_ij = a_ij - Σ_(k<j) l_ik·n_jk`
    /// for `i >= j`: each is ONE exact accumulation floored ONCE, the pivot is `d_j = n_jj`, and
    /// `l_ij = n_ij / d_j` is one truncated division. 6 divisions, no square root
    /// and no other product.
    ///
    /// `n_jk` IS the column of `l·diag(d)`: `l_jk·d_k` differs from it only by the remainder of
    /// the division that produced `l_jk`. Recomputing `l_jk·d_k` explicitly (the `alt_products`
    /// candidate of `benches.cairo`) costs 6 extra products, makes `new` 30 to 32 %
    /// dearer, and moves the factors FARTHER from the true ones: `a_jj - l_jk·(l_jk·d_k)` is
    /// second order in the rounding error of `l_jk` where `a_jj - l_jk·n_jk` is only first order
    /// (16 ulp against 34 at size 2, 27 against 42 at size 4). The backward error
    /// `l·diag(d)·lᵀ - a` is identical for both, being dominated by `l_ij·d_j - a_ij`, which
    /// no choice here changes — so the cheaper and more accurate form is the one that ships.
    ///
    /// SINGULARITY CRITERION: `None` iff a pivot `d_j` is EXACTLY zero, i.e. iff the leading
    /// principal minor of order j is zero after flooring. Unlike `Cholesky::new`, a negative
    /// pivot is accepted: `LDLᵀ` factorises any symmetric matrix whose leading principal minors
    /// are non-zero, indefinite ones included, and `d` then has mixed signs. It is NOT a
    /// definiteness test, and it has no pivoting: a symmetric matrix with a zero leading minor
    /// (`[[0, 1], [1, 0]]`) is rejected although it is invertible.
    ///
    /// Panics on overflow of a pivot or a numerator; never wraps.
    fn new(a: Matrix4<T>) -> Option<Ldlt4<T>> {
        let d1 = a.m11;
        if d1 == R::ZERO {
            return None;
        }
        let n21 = a.m21;
        let l21 = R::div(n21, d1);
        let n31 = a.m31;
        let l31 = R::div(n31, d1);
        let n41 = a.m41;
        let l41 = R::div(n41, d1);
        let w = R::wide_add(R::wide_zero(), a.m22);
        let w = R::wide_sub_prod(w, l21, n21);
        let d2 = R::wide_rescale(w);
        if d2 == R::ZERO {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m32);
        let w = R::wide_sub_prod(w, l31, n21);
        let n32 = R::wide_rescale(w);
        let l32 = R::div(n32, d2);
        let w = R::wide_add(R::wide_zero(), a.m42);
        let w = R::wide_sub_prod(w, l41, n21);
        let n42 = R::wide_rescale(w);
        let l42 = R::div(n42, d2);
        let w = R::wide_add(R::wide_zero(), a.m33);
        let w = R::wide_sub_prod(w, l31, n31);
        let w = R::wide_sub_prod(w, l32, n32);
        let d3 = R::wide_rescale(w);
        if d3 == R::ZERO {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m43);
        let w = R::wide_sub_prod(w, l41, n31);
        let w = R::wide_sub_prod(w, l42, n32);
        let n43 = R::wide_rescale(w);
        let l43 = R::div(n43, d3);
        let w = R::wide_add(R::wide_zero(), a.m44);
        let w = R::wide_sub_prod(w, l41, n41);
        let w = R::wide_sub_prod(w, l42, n42);
        let w = R::wide_sub_prod(w, l43, n43);
        let d4 = R::wide_rescale(w);
        if d4 == R::ZERO {
            return None;
        }
        Some(Ldlt4 { l21, l31, l41, l32, l42, l43, d: Vector4 { x: d1, y: d2, z: d3, w: d4 } })
    }

    /// The unit lower triangular factor, with explicit ones on the diagonal and zeros above it.
    /// Upstream: the reverse-permuted `UDU::u`. Exact: nothing is recomputed.
    #[inline(always)]
    fn l(self: Ldlt4<T>) -> Matrix4<T> {
        Matrix4 {
            m11: R::ONE,
            m21: self.l21,
            m31: self.l31,
            m41: self.l41,
            m12: R::ZERO,
            m22: R::ONE,
            m32: self.l32,
            m42: self.l42,
            m13: R::ZERO,
            m23: R::ZERO,
            m33: R::ONE,
            m43: self.l43,
            m14: R::ZERO,
            m24: R::ZERO,
            m34: R::ZERO,
            m44: R::ONE,
        }
    }

    /// The diagonal of `diag(d)`. Upstream: `UDU::d`, reversed. Exact: the stored vector.
    #[inline(always)]
    fn d(self: Ldlt4<T>) -> Vector4<T> {
        self.d
    }

    /// The solution `x` of `a * x = b`: forward substitution on the unit `l`, division by `d`,
    /// back substitution on the unit `lᵀ`. Upstream: `UDU` has no `solve`; the oracle's
    /// `udu4_solve` vectors apply unchanged.
    ///
    /// The unit diagonal removes the divisions of the two substitutions, so this costs 4
    /// divisions where `Cholesky4::solve` costs 8: each of the 8 substitution
    /// intermediates is ONE exact accumulation floored once and nothing more, and only the
    /// diagonal solve `z_i = y_i / d_i` divides.
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Ldlt4<T>, b: Vector4<T>) -> Vector4<T> {
        let y1 = b.x;
        let w = R::wide_add(R::wide_zero(), b.y);
        let w = R::wide_sub_prod(w, self.l21, y1);
        let y2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.z);
        let w = R::wide_sub_prod(w, self.l31, y1);
        let w = R::wide_sub_prod(w, self.l32, y2);
        let y3 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.w);
        let w = R::wide_sub_prod(w, self.l41, y1);
        let w = R::wide_sub_prod(w, self.l42, y2);
        let w = R::wide_sub_prod(w, self.l43, y3);
        let y4 = R::wide_rescale(w);
        let z1 = R::div(y1, self.d.x);
        let z2 = R::div(y2, self.d.y);
        let z3 = R::div(y3, self.d.z);
        let z4 = R::div(y4, self.d.w);
        let x4 = z4;
        let w = R::wide_add(R::wide_zero(), z3);
        let w = R::wide_sub_prod(w, self.l43, x4);
        let x3 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z2);
        let w = R::wide_sub_prod(w, self.l32, x3);
        let w = R::wide_sub_prod(w, self.l42, x4);
        let x2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let w = R::wide_sub_prod(w, self.l31, x3);
        let w = R::wide_sub_prod(w, self.l41, x4);
        let x1 = R::wide_rescale(w);
        Vector4 { x: x1, y: x2, z: x3, w: x4 }
    }

    /// `a⁻¹ = l⁻ᵀ · diag(d)⁻¹ · l⁻¹`, as a full (symmetric) `Matrix4`.
    ///
    /// `q = l⁻¹` is unit lower triangular and is built WITHOUT any division
    /// (`q_ij = -(l_ij + Σ_(k=j+1..i-1) l_ik·q_kj)`, one exact accumulation floored once); the
    /// 10 scaled entries `s_kj = q_kj / d_k` are the only divisions (4 of them are
    /// `recip(d_k)`, the `1 / d_k` truncated toward zero), and `a⁻¹_ij = Σ_k q_ki·s_kj` is one
    /// exact accumulation floored once per output component.
    ///
    /// The result is symmetric by construction, so only its upper triangle is computed — the sum
    /// is taken as `Σ_k q_ki·s_kj` with `i <= j`, which for rounded `s` differs from
    /// `Σ_k q_kj·s_ki` by at most a few ulp. Panics on overflow.
    fn inverse(self: Ldlt4<T>) -> Matrix4<T> {
        let q21 = -self.l21;
        let w = R::wide_sub(R::wide_zero(), self.l31);
        let w = R::wide_sub_prod(w, self.l32, q21);
        let q31 = R::wide_rescale(w);
        let w = R::wide_sub(R::wide_zero(), self.l41);
        let w = R::wide_sub_prod(w, self.l42, q21);
        let w = R::wide_sub_prod(w, self.l43, q31);
        let q41 = R::wide_rescale(w);
        let q32 = -self.l32;
        let w = R::wide_sub(R::wide_zero(), self.l42);
        let w = R::wide_sub_prod(w, self.l43, q32);
        let q42 = R::wide_rescale(w);
        let q43 = -self.l43;
        let s11 = R::recip(self.d.x);
        let s21 = R::div(q21, self.d.y);
        let s31 = R::div(q31, self.d.z);
        let s41 = R::div(q41, self.d.w);
        let s22 = R::recip(self.d.y);
        let s32 = R::div(q32, self.d.z);
        let s42 = R::div(q42, self.d.w);
        let s33 = R::recip(self.d.z);
        let s43 = R::div(q43, self.d.w);
        let s44 = R::recip(self.d.w);
        let w = R::wide_zero();
        let w = R::wide_add(w, s11);
        let w = R::wide_add_prod(w, q21, s21);
        let w = R::wide_add_prod(w, q31, s31);
        let w = R::wide_add_prod(w, q41, s41);
        let r11 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q21, s22);
        let w = R::wide_add_prod(w, q31, s32);
        let w = R::wide_add_prod(w, q41, s42);
        let r12 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q31, s33);
        let w = R::wide_add_prod(w, q41, s43);
        let r13 = R::wide_rescale(w);
        let r14 = q41 * s44;
        let w = R::wide_zero();
        let w = R::wide_add(w, s22);
        let w = R::wide_add_prod(w, q32, s32);
        let w = R::wide_add_prod(w, q42, s42);
        let r22 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q32, s33);
        let w = R::wide_add_prod(w, q42, s43);
        let r23 = R::wide_rescale(w);
        let r24 = q42 * s44;
        let w = R::wide_zero();
        let w = R::wide_add(w, s33);
        let w = R::wide_add_prod(w, q43, s43);
        let r33 = R::wide_rescale(w);
        let r34 = q43 * s44;
        let r44 = s44;
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

    /// `det(a) = Π d_j`, as a balanced product tree: 3 floored products, the
    /// shallowest chain of roundings for 4 factors. `det(l) = 1`, so no other term appears.
    ///
    /// Panics if the determinant does not fit the scalar. Underflow is silent (a determinant below
    /// 1 ulp floors to 0), which is the scalar's floor rounding and NOT a singularity report.
    #[inline(always)]
    fn determinant(self: Ldlt4<T>) -> T {
        (self.d.x * self.d.y) * (self.d.z * self.d.w)
    }
}

/// Methods of `Ldlt6<T>` for any `Real` scalar.
#[generate_trait]
pub impl Ldlt6Impl<
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
> of Ldlt6Trait<T> {
    /// The `LDLᵀ` factorisation `a = l * diag(d) * lᵀ` of the symmetric `a`, or `None` when a
    /// pivot is exactly zero. Upstream: `UDU::new` on the reversed matrix (see the module doc).
    ///
    /// Like upstream, only the LOWER triangle of `a` is read (the entries at row `i`,
    /// column `j` with `i >= j`): the strictly upper triangle is ignored and the symmetry
    /// of `a` is NOT checked. There is no `SymMatrix6` type, so a symmetric 6x6
    /// matrix travels as a plain `Matrix6`.
    ///
    /// Column by column (j = 1..6), in terms of the numerators `n_ij = a_ij - Σ_(k<j) l_ik·n_jk`
    /// for `i >= j`: each is ONE exact accumulation floored ONCE, the pivot is `d_j = n_jj`, and
    /// `l_ij = n_ij / d_j` is one truncated division. 15 divisions, no square root
    /// and no other product.
    ///
    /// `n_jk` IS the column of `l·diag(d)`: `l_jk·d_k` differs from it only by the remainder of
    /// the division that produced `l_jk`. Recomputing `l_jk·d_k` explicitly (the `alt_products`
    /// candidate of `benches.cairo`) costs 15 extra products, makes `new` 30 to 32 %
    /// dearer, and moves the factors FARTHER from the true ones: `a_jj - l_jk·(l_jk·d_k)` is
    /// second order in the rounding error of `l_jk` where `a_jj - l_jk·n_jk` is only first order
    /// (16 ulp against 34 at size 2, 27 against 42 at size 4). The backward error
    /// `l·diag(d)·lᵀ - a` is identical for both, being dominated by `l_ij·d_j - a_ij`, which
    /// no choice here changes — so the cheaper and more accurate form is the one that ships.
    ///
    /// SINGULARITY CRITERION: `None` iff a pivot `d_j` is EXACTLY zero, i.e. iff the leading
    /// principal minor of order j is zero after flooring. Unlike `Cholesky::new`, a negative
    /// pivot is accepted: `LDLᵀ` factorises any symmetric matrix whose leading principal minors
    /// are non-zero, indefinite ones included, and `d` then has mixed signs. It is NOT a
    /// definiteness test, and it has no pivoting: a symmetric matrix with a zero leading minor
    /// (`[[0, 1], [1, 0]]`) is rejected although it is invertible.
    ///
    /// Panics on overflow of a pivot or a numerator; never wraps.
    fn new(a: Matrix6<T>) -> Option<Ldlt6<T>> {
        let d1 = a.m11.m11;
        if d1 == R::ZERO {
            return None;
        }
        let n21 = a.m11.m21;
        let l21 = R::div(n21, d1);
        let n31 = a.m11.m31;
        let l31 = R::div(n31, d1);
        let n41 = a.m21.m11;
        let l41 = R::div(n41, d1);
        let n51 = a.m21.m21;
        let l51 = R::div(n51, d1);
        let n61 = a.m21.m31;
        let l61 = R::div(n61, d1);
        let w = R::wide_add(R::wide_zero(), a.m11.m22);
        let w = R::wide_sub_prod(w, l21, n21);
        let d2 = R::wide_rescale(w);
        if d2 == R::ZERO {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m11.m32);
        let w = R::wide_sub_prod(w, l31, n21);
        let n32 = R::wide_rescale(w);
        let l32 = R::div(n32, d2);
        let w = R::wide_add(R::wide_zero(), a.m21.m12);
        let w = R::wide_sub_prod(w, l41, n21);
        let n42 = R::wide_rescale(w);
        let l42 = R::div(n42, d2);
        let w = R::wide_add(R::wide_zero(), a.m21.m22);
        let w = R::wide_sub_prod(w, l51, n21);
        let n52 = R::wide_rescale(w);
        let l52 = R::div(n52, d2);
        let w = R::wide_add(R::wide_zero(), a.m21.m32);
        let w = R::wide_sub_prod(w, l61, n21);
        let n62 = R::wide_rescale(w);
        let l62 = R::div(n62, d2);
        let w = R::wide_add(R::wide_zero(), a.m11.m33);
        let w = R::wide_sub_prod(w, l31, n31);
        let w = R::wide_sub_prod(w, l32, n32);
        let d3 = R::wide_rescale(w);
        if d3 == R::ZERO {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m21.m13);
        let w = R::wide_sub_prod(w, l41, n31);
        let w = R::wide_sub_prod(w, l42, n32);
        let n43 = R::wide_rescale(w);
        let l43 = R::div(n43, d3);
        let w = R::wide_add(R::wide_zero(), a.m21.m23);
        let w = R::wide_sub_prod(w, l51, n31);
        let w = R::wide_sub_prod(w, l52, n32);
        let n53 = R::wide_rescale(w);
        let l53 = R::div(n53, d3);
        let w = R::wide_add(R::wide_zero(), a.m21.m33);
        let w = R::wide_sub_prod(w, l61, n31);
        let w = R::wide_sub_prod(w, l62, n32);
        let n63 = R::wide_rescale(w);
        let l63 = R::div(n63, d3);
        let w = R::wide_add(R::wide_zero(), a.m22.m11);
        let w = R::wide_sub_prod(w, l41, n41);
        let w = R::wide_sub_prod(w, l42, n42);
        let w = R::wide_sub_prod(w, l43, n43);
        let d4 = R::wide_rescale(w);
        if d4 == R::ZERO {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m22.m21);
        let w = R::wide_sub_prod(w, l51, n41);
        let w = R::wide_sub_prod(w, l52, n42);
        let w = R::wide_sub_prod(w, l53, n43);
        let n54 = R::wide_rescale(w);
        let l54 = R::div(n54, d4);
        let w = R::wide_add(R::wide_zero(), a.m22.m31);
        let w = R::wide_sub_prod(w, l61, n41);
        let w = R::wide_sub_prod(w, l62, n42);
        let w = R::wide_sub_prod(w, l63, n43);
        let n64 = R::wide_rescale(w);
        let l64 = R::div(n64, d4);
        let w = R::wide_add(R::wide_zero(), a.m22.m22);
        let w = R::wide_sub_prod(w, l51, n51);
        let w = R::wide_sub_prod(w, l52, n52);
        let w = R::wide_sub_prod(w, l53, n53);
        let w = R::wide_sub_prod(w, l54, n54);
        let d5 = R::wide_rescale(w);
        if d5 == R::ZERO {
            return None;
        }
        let w = R::wide_add(R::wide_zero(), a.m22.m32);
        let w = R::wide_sub_prod(w, l61, n51);
        let w = R::wide_sub_prod(w, l62, n52);
        let w = R::wide_sub_prod(w, l63, n53);
        let w = R::wide_sub_prod(w, l64, n54);
        let n65 = R::wide_rescale(w);
        let l65 = R::div(n65, d5);
        let w = R::wide_add(R::wide_zero(), a.m22.m33);
        let w = R::wide_sub_prod(w, l61, n61);
        let w = R::wide_sub_prod(w, l62, n62);
        let w = R::wide_sub_prod(w, l63, n63);
        let w = R::wide_sub_prod(w, l64, n64);
        let w = R::wide_sub_prod(w, l65, n65);
        let d6 = R::wide_rescale(w);
        if d6 == R::ZERO {
            return None;
        }
        Some(
            Ldlt6 {
                l21,
                l31,
                l41,
                l51,
                l61,
                l32,
                l42,
                l52,
                l62,
                l43,
                l53,
                l63,
                l54,
                l64,
                l65,
                d: Vector6 {
                    a: Vector3 { x: d1, y: d2, z: d3 }, b: Vector3 { x: d4, y: d5, z: d6 },
                },
            },
        )
    }

    /// The unit lower triangular factor, with explicit ones on the diagonal and zeros above it.
    /// Upstream: the reverse-permuted `UDU::u`. Exact: nothing is recomputed.
    #[inline(always)]
    fn l(self: Ldlt6<T>) -> Matrix6<T> {
        Matrix6 {
            m11: Matrix3 {
                m11: R::ONE,
                m21: self.l21,
                m31: self.l31,
                m12: R::ZERO,
                m22: R::ONE,
                m32: self.l32,
                m13: R::ZERO,
                m23: R::ZERO,
                m33: R::ONE,
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
                m11: R::ZERO,
                m21: R::ZERO,
                m31: R::ZERO,
                m12: R::ZERO,
                m22: R::ZERO,
                m32: R::ZERO,
                m13: R::ZERO,
                m23: R::ZERO,
                m33: R::ZERO,
            },
            m22: Matrix3 {
                m11: R::ONE,
                m21: self.l54,
                m31: self.l64,
                m12: R::ZERO,
                m22: R::ONE,
                m32: self.l65,
                m13: R::ZERO,
                m23: R::ZERO,
                m33: R::ONE,
            },
        }
    }

    /// The diagonal of `diag(d)`. Upstream: `UDU::d`, reversed. Exact: the stored vector.
    #[inline(always)]
    fn d(self: Ldlt6<T>) -> Vector6<T> {
        self.d
    }

    /// The solution `x` of `a * x = b`: forward substitution on the unit `l`, division by `d`,
    /// back substitution on the unit `lᵀ`. Upstream: `UDU` has no `solve`; the oracle's
    /// `udu6_solve` vectors apply unchanged.
    ///
    /// The unit diagonal removes the divisions of the two substitutions, so this costs 6
    /// divisions where `Cholesky6::solve` costs 12: each of the 12 substitution
    /// intermediates is ONE exact accumulation floored once and nothing more, and only the
    /// diagonal solve `z_i = y_i / d_i` divides.
    ///
    /// Panics on overflow. A factor built by `new` has non-zero pivots, so no division by zero can
    /// occur; a hand-assembled factor with a zero pivot panics with the scalar's error.
    fn solve(self: Ldlt6<T>, b: Vector6<T>) -> Vector6<T> {
        let y1 = b.a.x;
        let w = R::wide_add(R::wide_zero(), b.a.y);
        let w = R::wide_sub_prod(w, self.l21, y1);
        let y2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.a.z);
        let w = R::wide_sub_prod(w, self.l31, y1);
        let w = R::wide_sub_prod(w, self.l32, y2);
        let y3 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.b.x);
        let w = R::wide_sub_prod(w, self.l41, y1);
        let w = R::wide_sub_prod(w, self.l42, y2);
        let w = R::wide_sub_prod(w, self.l43, y3);
        let y4 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.b.y);
        let w = R::wide_sub_prod(w, self.l51, y1);
        let w = R::wide_sub_prod(w, self.l52, y2);
        let w = R::wide_sub_prod(w, self.l53, y3);
        let w = R::wide_sub_prod(w, self.l54, y4);
        let y5 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), b.b.z);
        let w = R::wide_sub_prod(w, self.l61, y1);
        let w = R::wide_sub_prod(w, self.l62, y2);
        let w = R::wide_sub_prod(w, self.l63, y3);
        let w = R::wide_sub_prod(w, self.l64, y4);
        let w = R::wide_sub_prod(w, self.l65, y5);
        let y6 = R::wide_rescale(w);
        let z1 = R::div(y1, self.d.a.x);
        let z2 = R::div(y2, self.d.a.y);
        let z3 = R::div(y3, self.d.a.z);
        let z4 = R::div(y4, self.d.b.x);
        let z5 = R::div(y5, self.d.b.y);
        let z6 = R::div(y6, self.d.b.z);
        let x6 = z6;
        let w = R::wide_add(R::wide_zero(), z5);
        let w = R::wide_sub_prod(w, self.l65, x6);
        let x5 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z4);
        let w = R::wide_sub_prod(w, self.l54, x5);
        let w = R::wide_sub_prod(w, self.l64, x6);
        let x4 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z3);
        let w = R::wide_sub_prod(w, self.l43, x4);
        let w = R::wide_sub_prod(w, self.l53, x5);
        let w = R::wide_sub_prod(w, self.l63, x6);
        let x3 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z2);
        let w = R::wide_sub_prod(w, self.l32, x3);
        let w = R::wide_sub_prod(w, self.l42, x4);
        let w = R::wide_sub_prod(w, self.l52, x5);
        let w = R::wide_sub_prod(w, self.l62, x6);
        let x2 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), z1);
        let w = R::wide_sub_prod(w, self.l21, x2);
        let w = R::wide_sub_prod(w, self.l31, x3);
        let w = R::wide_sub_prod(w, self.l41, x4);
        let w = R::wide_sub_prod(w, self.l51, x5);
        let w = R::wide_sub_prod(w, self.l61, x6);
        let x1 = R::wide_rescale(w);
        Vector6 { a: Vector3 { x: x1, y: x2, z: x3 }, b: Vector3 { x: x4, y: x5, z: x6 } }
    }

    /// `a⁻¹ = l⁻ᵀ · diag(d)⁻¹ · l⁻¹`, as a full (symmetric) `Matrix6`.
    ///
    /// `q = l⁻¹` is unit lower triangular and is built WITHOUT any division
    /// (`q_ij = -(l_ij + Σ_(k=j+1..i-1) l_ik·q_kj)`, one exact accumulation floored once); the
    /// 21 scaled entries `s_kj = q_kj / d_k` are the only divisions (6 of them are
    /// `recip(d_k)`, the `1 / d_k` truncated toward zero), and `a⁻¹_ij = Σ_k q_ki·s_kj` is one
    /// exact accumulation floored once per output component.
    ///
    /// The result is symmetric by construction, so only its upper triangle is computed — the sum
    /// is taken as `Σ_k q_ki·s_kj` with `i <= j`, which for rounded `s` differs from
    /// `Σ_k q_kj·s_ki` by at most a few ulp. Panics on overflow.
    fn inverse(self: Ldlt6<T>) -> Matrix6<T> {
        let q21 = -self.l21;
        let w = R::wide_sub(R::wide_zero(), self.l31);
        let w = R::wide_sub_prod(w, self.l32, q21);
        let q31 = R::wide_rescale(w);
        let w = R::wide_sub(R::wide_zero(), self.l41);
        let w = R::wide_sub_prod(w, self.l42, q21);
        let w = R::wide_sub_prod(w, self.l43, q31);
        let q41 = R::wide_rescale(w);
        let w = R::wide_sub(R::wide_zero(), self.l51);
        let w = R::wide_sub_prod(w, self.l52, q21);
        let w = R::wide_sub_prod(w, self.l53, q31);
        let w = R::wide_sub_prod(w, self.l54, q41);
        let q51 = R::wide_rescale(w);
        let w = R::wide_sub(R::wide_zero(), self.l61);
        let w = R::wide_sub_prod(w, self.l62, q21);
        let w = R::wide_sub_prod(w, self.l63, q31);
        let w = R::wide_sub_prod(w, self.l64, q41);
        let w = R::wide_sub_prod(w, self.l65, q51);
        let q61 = R::wide_rescale(w);
        let q32 = -self.l32;
        let w = R::wide_sub(R::wide_zero(), self.l42);
        let w = R::wide_sub_prod(w, self.l43, q32);
        let q42 = R::wide_rescale(w);
        let w = R::wide_sub(R::wide_zero(), self.l52);
        let w = R::wide_sub_prod(w, self.l53, q32);
        let w = R::wide_sub_prod(w, self.l54, q42);
        let q52 = R::wide_rescale(w);
        let w = R::wide_sub(R::wide_zero(), self.l62);
        let w = R::wide_sub_prod(w, self.l63, q32);
        let w = R::wide_sub_prod(w, self.l64, q42);
        let w = R::wide_sub_prod(w, self.l65, q52);
        let q62 = R::wide_rescale(w);
        let q43 = -self.l43;
        let w = R::wide_sub(R::wide_zero(), self.l53);
        let w = R::wide_sub_prod(w, self.l54, q43);
        let q53 = R::wide_rescale(w);
        let w = R::wide_sub(R::wide_zero(), self.l63);
        let w = R::wide_sub_prod(w, self.l64, q43);
        let w = R::wide_sub_prod(w, self.l65, q53);
        let q63 = R::wide_rescale(w);
        let q54 = -self.l54;
        let w = R::wide_sub(R::wide_zero(), self.l64);
        let w = R::wide_sub_prod(w, self.l65, q54);
        let q64 = R::wide_rescale(w);
        let q65 = -self.l65;
        let s11 = R::recip(self.d.a.x);
        let s21 = R::div(q21, self.d.a.y);
        let s31 = R::div(q31, self.d.a.z);
        let s41 = R::div(q41, self.d.b.x);
        let s51 = R::div(q51, self.d.b.y);
        let s61 = R::div(q61, self.d.b.z);
        let s22 = R::recip(self.d.a.y);
        let s32 = R::div(q32, self.d.a.z);
        let s42 = R::div(q42, self.d.b.x);
        let s52 = R::div(q52, self.d.b.y);
        let s62 = R::div(q62, self.d.b.z);
        let s33 = R::recip(self.d.a.z);
        let s43 = R::div(q43, self.d.b.x);
        let s53 = R::div(q53, self.d.b.y);
        let s63 = R::div(q63, self.d.b.z);
        let s44 = R::recip(self.d.b.x);
        let s54 = R::div(q54, self.d.b.y);
        let s64 = R::div(q64, self.d.b.z);
        let s55 = R::recip(self.d.b.y);
        let s65 = R::div(q65, self.d.b.z);
        let s66 = R::recip(self.d.b.z);
        let w = R::wide_zero();
        let w = R::wide_add(w, s11);
        let w = R::wide_add_prod(w, q21, s21);
        let w = R::wide_add_prod(w, q31, s31);
        let w = R::wide_add_prod(w, q41, s41);
        let w = R::wide_add_prod(w, q51, s51);
        let w = R::wide_add_prod(w, q61, s61);
        let r11 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q21, s22);
        let w = R::wide_add_prod(w, q31, s32);
        let w = R::wide_add_prod(w, q41, s42);
        let w = R::wide_add_prod(w, q51, s52);
        let w = R::wide_add_prod(w, q61, s62);
        let r12 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q31, s33);
        let w = R::wide_add_prod(w, q41, s43);
        let w = R::wide_add_prod(w, q51, s53);
        let w = R::wide_add_prod(w, q61, s63);
        let r13 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q41, s44);
        let w = R::wide_add_prod(w, q51, s54);
        let w = R::wide_add_prod(w, q61, s64);
        let r14 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q51, s55);
        let w = R::wide_add_prod(w, q61, s65);
        let r15 = R::wide_rescale(w);
        let r16 = q61 * s66;
        let w = R::wide_zero();
        let w = R::wide_add(w, s22);
        let w = R::wide_add_prod(w, q32, s32);
        let w = R::wide_add_prod(w, q42, s42);
        let w = R::wide_add_prod(w, q52, s52);
        let w = R::wide_add_prod(w, q62, s62);
        let r22 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q32, s33);
        let w = R::wide_add_prod(w, q42, s43);
        let w = R::wide_add_prod(w, q52, s53);
        let w = R::wide_add_prod(w, q62, s63);
        let r23 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q42, s44);
        let w = R::wide_add_prod(w, q52, s54);
        let w = R::wide_add_prod(w, q62, s64);
        let r24 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q52, s55);
        let w = R::wide_add_prod(w, q62, s65);
        let r25 = R::wide_rescale(w);
        let r26 = q62 * s66;
        let w = R::wide_zero();
        let w = R::wide_add(w, s33);
        let w = R::wide_add_prod(w, q43, s43);
        let w = R::wide_add_prod(w, q53, s53);
        let w = R::wide_add_prod(w, q63, s63);
        let r33 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q43, s44);
        let w = R::wide_add_prod(w, q53, s54);
        let w = R::wide_add_prod(w, q63, s64);
        let r34 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q53, s55);
        let w = R::wide_add_prod(w, q63, s65);
        let r35 = R::wide_rescale(w);
        let r36 = q63 * s66;
        let w = R::wide_zero();
        let w = R::wide_add(w, s44);
        let w = R::wide_add_prod(w, q54, s54);
        let w = R::wide_add_prod(w, q64, s64);
        let r44 = R::wide_rescale(w);
        let w = R::wide_zero();
        let w = R::wide_add_prod(w, q54, s55);
        let w = R::wide_add_prod(w, q64, s65);
        let r45 = R::wide_rescale(w);
        let r46 = q64 * s66;
        let w = R::wide_zero();
        let w = R::wide_add(w, s55);
        let w = R::wide_add_prod(w, q65, s65);
        let r55 = R::wide_rescale(w);
        let r56 = q65 * s66;
        let r66 = s66;
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

    /// `det(a) = Π d_j`, as a balanced product tree: 5 floored products, the
    /// shallowest chain of roundings for 6 factors. `det(l) = 1`, so no other term appears.
    ///
    /// Panics if the determinant does not fit the scalar. Underflow is silent (a determinant below
    /// 1 ulp floors to 0), which is the scalar's floor rounding and NOT a singularity report.
    #[inline(always)]
    fn determinant(self: Ldlt6<T>) -> T {
        (self.d.a.x * (self.d.a.y * self.d.a.z)) * (self.d.b.x * (self.d.b.y * self.d.b.z))
    }
}
