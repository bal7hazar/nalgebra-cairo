//! In `nalgebra_linalg6`: the `struct` `Cholesky6`; its 2 impls, among them `Cholesky6Impl`,
//! `Matrix6CholeskyImpl`. This module is split over packages; the other parts are in
//! `nalgebra_linalg2`, `nalgebra_linalg3`, `nalgebra_linalg4`.
//!
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
use nalgebra_types6::base::matrix6::Matrix6;
use nalgebra_types6::base::vector6::Vector6;
use simba::scalar::{Real, Transcendental};

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
    /// 6 square roots and 15 divisions in total; the quotients of columns 1, 2 and 3 share one
    /// prepared divisor per column (`Real::div5`, `div4`, `div3`: bit-identical to per-element
    /// division, WP 11-OPT-2).
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
    fn new(a: Matrix6<T>) -> Option<Cholesky6<T>> {
        let p1 = a.m11;
        if p1 <= R::zero() {
            return None;
        }
        let l11 = R::sqrt(p1);
        let (l21, l31, l41, l51, l61) = R::div5(a.m21, a.m31, a.m41, a.m51, a.m61, l11);
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
        let w = R::wide_add(R::wide_zero(), a.m42);
        let w = R::wide_sub_prod(w, l41, l21);
        let n42 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), a.m52);
        let w = R::wide_sub_prod(w, l51, l21);
        let n52 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), a.m62);
        let w = R::wide_sub_prod(w, l61, l21);
        let n62 = R::wide_rescale(w);
        let (l32, l42, l52, l62) = R::div4(n32, n42, n52, n62, l22);
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
        let w = R::wide_add(R::wide_zero(), a.m53);
        let w = R::wide_sub_prod(w, l51, l31);
        let w = R::wide_sub_prod(w, l52, l32);
        let n53 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), a.m63);
        let w = R::wide_sub_prod(w, l61, l31);
        let w = R::wide_sub_prod(w, l62, l32);
        let n63 = R::wide_rescale(w);
        let (l43, l53, l63) = R::div3(n43, n53, n63, l33);
        let w = R::wide_add(R::wide_zero(), a.m44);
        let w = R::wide_sub_prod(w, l41, l41);
        let w = R::wide_sub_prod(w, l42, l42);
        let w = R::wide_sub_prod(w, l43, l43);
        let p4 = R::wide_rescale(w);
        if p4 <= R::zero() {
            return None;
        }
        let l44 = R::sqrt(p4);
        let w = R::wide_add(R::wide_zero(), a.m54);
        let w = R::wide_sub_prod(w, l51, l41);
        let w = R::wide_sub_prod(w, l52, l42);
        let w = R::wide_sub_prod(w, l53, l43);
        let n54 = R::wide_rescale(w);
        let l54 = R::div(n54, l44);
        let w = R::wide_add(R::wide_zero(), a.m64);
        let w = R::wide_sub_prod(w, l61, l41);
        let w = R::wide_sub_prod(w, l62, l42);
        let w = R::wide_sub_prod(w, l63, l43);
        let n64 = R::wide_rescale(w);
        let l64 = R::div(n64, l44);
        let w = R::wide_add(R::wide_zero(), a.m55);
        let w = R::wide_sub_prod(w, l51, l51);
        let w = R::wide_sub_prod(w, l52, l52);
        let w = R::wide_sub_prod(w, l53, l53);
        let w = R::wide_sub_prod(w, l54, l54);
        let p5 = R::wide_rescale(w);
        if p5 <= R::zero() {
            return None;
        }
        let l55 = R::sqrt(p5);
        let w = R::wide_add(R::wide_zero(), a.m65);
        let w = R::wide_sub_prod(w, l61, l51);
        let w = R::wide_sub_prod(w, l62, l52);
        let w = R::wide_sub_prod(w, l63, l53);
        let w = R::wide_sub_prod(w, l64, l54);
        let n65 = R::wide_rescale(w);
        let l65 = R::div(n65, l55);
        let w = R::wide_add(R::wide_zero(), a.m66);
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
            m11: self.l11,
            m21: self.l21,
            m31: self.l31,
            m12: R::zero(),
            m22: self.l22,
            m32: self.l32,
            m13: R::zero(),
            m23: R::zero(),
            m33: self.l33,
            m41: self.l41,
            m51: self.l51,
            m61: self.l61,
            m42: self.l42,
            m52: self.l52,
            m62: self.l62,
            m43: self.l43,
            m53: self.l53,
            m63: self.l63,
            m14: R::zero(),
            m24: R::zero(),
            m34: R::zero(),
            m15: R::zero(),
            m25: R::zero(),
            m35: R::zero(),
            m16: R::zero(),
            m26: R::zero(),
            m36: R::zero(),
            m44: self.l44,
            m54: self.l54,
            m64: self.l64,
            m45: R::zero(),
            m55: self.l55,
            m65: self.l65,
            m46: R::zero(),
            m56: R::zero(),
            m66: self.l66,
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
    #[inline(always)]
    fn solve(self: Cholesky6<T>, b: Vector6<T>) -> Vector6<T> {
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
        let w = R::wide_add(R::wide_zero(), b.a);
        let w = R::wide_sub_prod(w, self.l51, y1);
        let w = R::wide_sub_prod(w, self.l52, y2);
        let w = R::wide_sub_prod(w, self.l53, y3);
        let w = R::wide_sub_prod(w, self.l54, y4);
        let f5 = R::wide_rescale(w);
        let y5 = R::div(f5, self.l55);
        let w = R::wide_add(R::wide_zero(), b.b);
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
        Vector6 { x: x1, y: x2, z: x3, w: x4, a: x5, b: x6 }
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
            m11: r11,
            m21: r12,
            m31: r13,
            m12: r12,
            m22: r22,
            m32: r23,
            m13: r13,
            m23: r23,
            m33: r33,
            m41: r14,
            m51: r15,
            m61: r16,
            m42: r24,
            m52: r25,
            m62: r26,
            m43: r34,
            m53: r35,
            m63: r36,
            m14: r14,
            m24: r24,
            m34: r34,
            m15: r15,
            m25: r25,
            m35: r35,
            m16: r16,
            m26: r26,
            m36: r36,
            m44: r44,
            m54: r45,
            m64: r46,
            m45: r45,
            m55: r55,
            m65: r56,
            m46: r46,
            m56: r56,
            m66: r66,
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

    /// The factorisation computed WITHOUT the positivity test, like upstream's
    /// `Cholesky::new_unchecked` — except that a pivot `<= 0` (upstream: the square root of a
    /// negative number or a division by zero, NaN / infinities, which a fixed-point scalar does
    /// not have) panics with `nalgebra: not positive definite`. Same kernel, same rounding and
    /// same gas as `new`. Upstream: `Cholesky::new_unchecked`.
    #[inline(always)]
    fn new_unchecked(matrix: Matrix6<T>) -> Cholesky6<T> {
        Self::new(matrix).expect(NOT_POSITIVE_DEFINITE)
    }

    /// `new` with `substitute` in place of every pivot that is `<= 0` after flooring (upstream:
    /// zero or negative, where the square root fails); `None` when such a pivot occurs and
    /// `substitute` is itself `<= 0`. The same kernel as `new` otherwise (same rounding). Upstream:
    /// `Cholesky::new_with_substitute`.
    fn new_with_substitute(a: Matrix6<T>, substitute: T) -> Option<Cholesky6<T>> {
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
        let (l21, l31, l41, l51, l61) = R::div5(a.m21, a.m31, a.m41, a.m51, a.m61, l11);
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
        let w = R::wide_add(R::wide_zero(), a.m32);
        let w = R::wide_sub_prod(w, l31, l21);
        let n32 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), a.m42);
        let w = R::wide_sub_prod(w, l41, l21);
        let n42 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), a.m52);
        let w = R::wide_sub_prod(w, l51, l21);
        let n52 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), a.m62);
        let w = R::wide_sub_prod(w, l61, l21);
        let n62 = R::wide_rescale(w);
        let (l32, l42, l52, l62) = R::div4(n32, n42, n52, n62, l22);
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
        let w = R::wide_add(R::wide_zero(), a.m43);
        let w = R::wide_sub_prod(w, l41, l31);
        let w = R::wide_sub_prod(w, l42, l32);
        let n43 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), a.m53);
        let w = R::wide_sub_prod(w, l51, l31);
        let w = R::wide_sub_prod(w, l52, l32);
        let n53 = R::wide_rescale(w);
        let w = R::wide_add(R::wide_zero(), a.m63);
        let w = R::wide_sub_prod(w, l61, l31);
        let w = R::wide_sub_prod(w, l62, l32);
        let n63 = R::wide_rescale(w);
        let (l43, l53, l63) = R::div3(n43, n53, n63, l33);
        let w = R::wide_add(R::wide_zero(), a.m44);
        let w = R::wide_sub_prod(w, l41, l41);
        let w = R::wide_sub_prod(w, l42, l42);
        let w = R::wide_sub_prod(w, l43, l43);
        let p4 = R::wide_rescale(w);
        let p4 = if p4 <= R::zero() {
            substitute
        } else {
            p4
        };
        if p4 <= R::zero() {
            return None;
        }
        let l44 = R::sqrt(p4);
        let w = R::wide_add(R::wide_zero(), a.m54);
        let w = R::wide_sub_prod(w, l51, l41);
        let w = R::wide_sub_prod(w, l52, l42);
        let w = R::wide_sub_prod(w, l53, l43);
        let n54 = R::wide_rescale(w);
        let l54 = R::div(n54, l44);
        let w = R::wide_add(R::wide_zero(), a.m64);
        let w = R::wide_sub_prod(w, l61, l41);
        let w = R::wide_sub_prod(w, l62, l42);
        let w = R::wide_sub_prod(w, l63, l43);
        let n64 = R::wide_rescale(w);
        let l64 = R::div(n64, l44);
        let w = R::wide_add(R::wide_zero(), a.m55);
        let w = R::wide_sub_prod(w, l51, l51);
        let w = R::wide_sub_prod(w, l52, l52);
        let w = R::wide_sub_prod(w, l53, l53);
        let w = R::wide_sub_prod(w, l54, l54);
        let p5 = R::wide_rescale(w);
        let p5 = if p5 <= R::zero() {
            substitute
        } else {
            p5
        };
        if p5 <= R::zero() {
            return None;
        }
        let l55 = R::sqrt(p5);
        let w = R::wide_add(R::wide_zero(), a.m65);
        let w = R::wide_sub_prod(w, l61, l51);
        let w = R::wide_sub_prod(w, l62, l52);
        let w = R::wide_sub_prod(w, l63, l53);
        let w = R::wide_sub_prod(w, l64, l54);
        let n65 = R::wide_rescale(w);
        let l65 = R::div(n65, l55);
        let w = R::wide_add(R::wide_zero(), a.m66);
        let w = R::wide_sub_prod(w, l61, l61);
        let w = R::wide_sub_prod(w, l62, l62);
        let w = R::wide_sub_prod(w, l63, l63);
        let w = R::wide_sub_prod(w, l64, l64);
        let w = R::wide_sub_prod(w, l65, l65);
        let p6 = R::wide_rescale(w);
        let p6 = if p6 <= R::zero() {
            substitute
        } else {
            p6
        };
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

    /// The factor `l` from the LOWER triangle of `matrix` (the strictly upper triangle is not
    /// read, like upstream's "dirty" storage). Nothing is checked. Exact: moves. Upstream:
    /// `Cholesky::pack_dirty`.
    #[inline(always)]
    fn pack_dirty(matrix: Matrix6<T>) -> Cholesky6<T> {
        Cholesky6 {
            l11: matrix.m11,
            l21: matrix.m21,
            l31: matrix.m31,
            l41: matrix.m41,
            l51: matrix.m51,
            l61: matrix.m61,
            l22: matrix.m22,
            l32: matrix.m32,
            l42: matrix.m42,
            l52: matrix.m52,
            l62: matrix.m62,
            l33: matrix.m33,
            l43: matrix.m43,
            l53: matrix.m53,
            l63: matrix.m63,
            l44: matrix.m44,
            l54: matrix.m54,
            l64: matrix.m64,
            l55: matrix.m55,
            l65: matrix.m65,
            l66: matrix.m66,
        }
    }

    /// The lower triangular factor, consuming the factorisation: `l()`. Exact. Upstream:
    /// `Cholesky::unpack`.
    #[inline(always)]
    fn unpack(self: Cholesky6<T>) -> Matrix6<T> {
        Self::l(self)
    }

    /// The factor with its "dirty" upper triangle: `l()` here, the compact storage keeps no upper
    /// triangle (upstream returns the input's upper triangle there, unspecified by its docs).
    /// Exact.
    /// Upstream: `Cholesky::unpack_dirty`.
    #[inline(always)]
    fn unpack_dirty(self: Cholesky6<T>) -> Matrix6<T> {
        Self::l(self)
    }

    /// The factor with its "dirty" upper triangle, by value: `l()` (see `unpack_dirty`). Exact.
    /// Upstream: `Cholesky::l_dirty` (a reference to the storage).
    #[inline(always)]
    fn l_dirty(self: Cholesky6<T>) -> Matrix6<T> {
        Self::l(self)
    }

    /// Overwrites `b` (any shape with 6 rows: a vector or a matrix) with the solution of `a * x =
    /// b`: `l y = b` by forward substitution then `lᵀ x = y` by back substitution, each component
    /// ONE exact sum floored once then one correctly rounded division by the pivot (one prepared
    /// divisor per row from 3 columns on). On a vector it is bit-identical to `solve`. Panics on
    /// overflow. Upstream: `Cholesky::solve_mut`.
    fn solve_mut<B, impl K: SolveKernel<Matrix6<T>, B>, +Drop<B>>(self: Cholesky6<T>, ref b: B) {
        let lt = Matrix6 {
            m11: self.l11,
            m21: R::zero(),
            m31: R::zero(),
            m41: R::zero(),
            m51: R::zero(),
            m61: R::zero(),
            m12: self.l21,
            m22: self.l22,
            m32: R::zero(),
            m42: R::zero(),
            m52: R::zero(),
            m62: R::zero(),
            m13: self.l31,
            m23: self.l32,
            m33: self.l33,
            m43: R::zero(),
            m53: R::zero(),
            m63: R::zero(),
            m14: self.l41,
            m24: self.l42,
            m34: self.l43,
            m44: self.l44,
            m54: R::zero(),
            m64: R::zero(),
            m15: self.l51,
            m25: self.l52,
            m35: self.l53,
            m45: self.l54,
            m55: self.l55,
            m65: R::zero(),
            m16: self.l61,
            m26: self.l62,
            m36: self.l63,
            m46: self.l64,
            m56: self.l65,
            m66: self.l66,
        };
        b = K::upper(lt, K::lower(Self::l(self), b));
    }

    /// `ln(det(a)) = Σ ln(l_jj²)`, computed as `2 Σ ln(l_jj)`: 6 natural logarithms, an exact
    /// sum and an exact doubling. Upstream sums `ln(l_jj²)`; squaring first would floor `l_jj²`
    /// (and send a pivot below 2^-16 to `ln(0)`), so the logarithm of the pivot itself is both
    /// cheaper and more accurate. Needs `Transcendental`. Upstream: `Cholesky::ln_determinant`.
    fn ln_determinant<+Transcendental<T>>(self: Cholesky6<T>) -> T {
        let s = Transcendental::ln(self.l11)
            + Transcendental::ln(self.l22)
            + Transcendental::ln(self.l33)
            + Transcendental::ln(self.l44)
            + Transcendental::ln(self.l55)
            + Transcendental::ln(self.l66);
        s + s
    }
}

/// `SquareMatrix::cholesky` on `Matrix6<T>` (upstream `nalgebra::linalg` decomposition entry
/// point).
#[generate_trait]
pub impl Matrix6CholeskyImpl<
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
> of Matrix6CholeskyTrait<T> {
    /// The Cholesky factorisation of `self` (its lower triangle), or `None` when it is not
    /// positive definite: `Cholesky6Trait::new(self)`. Upstream: `SquareMatrix::cholesky`.
    #[inline(always)]
    fn cholesky(self: Matrix6<T>) -> Option<Cholesky6<T>> {
        Cholesky6Trait::new(self)
    }
}

#[cfg(test)]
mod tests {
    use fixed::Fixed;
    use nalgebra_types6::base::matrix6::Matrix6;
    use nalgebra_types6::base::vector6::Vector6;
    use simba::scalar::FixedReal as R;
    use super::{Cholesky6, Cholesky6Trait};

    /// `Cholesky6Trait::new` before WP 11-OPT-2: one division per sub-diagonal entry (the new body
    /// shares one prepared divisor per column, `Real::div4` / `div3`), called, not inlined.
    fn new_reference(a: Matrix6<Fixed>) -> Option<Cholesky6<Fixed>> {
        let p1 = a.m11;
        if p1 <= R::zero() {
            return None;
        }
        let l11 = R::sqrt(p1);
        let (l21, l31, l41, l51, l61) = R::div5(a.m21, a.m31, a.m41, a.m51, a.m61, l11);
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
        let w = R::wide_add(R::wide_zero(), a.m52);
        let w = R::wide_sub_prod(w, l51, l21);
        let n52 = R::wide_rescale(w);
        let l52 = R::div(n52, l22);
        let w = R::wide_add(R::wide_zero(), a.m62);
        let w = R::wide_sub_prod(w, l61, l21);
        let n62 = R::wide_rescale(w);
        let l62 = R::div(n62, l22);
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
        let w = R::wide_add(R::wide_zero(), a.m53);
        let w = R::wide_sub_prod(w, l51, l31);
        let w = R::wide_sub_prod(w, l52, l32);
        let n53 = R::wide_rescale(w);
        let l53 = R::div(n53, l33);
        let w = R::wide_add(R::wide_zero(), a.m63);
        let w = R::wide_sub_prod(w, l61, l31);
        let w = R::wide_sub_prod(w, l62, l32);
        let n63 = R::wide_rescale(w);
        let l63 = R::div(n63, l33);
        let w = R::wide_add(R::wide_zero(), a.m44);
        let w = R::wide_sub_prod(w, l41, l41);
        let w = R::wide_sub_prod(w, l42, l42);
        let w = R::wide_sub_prod(w, l43, l43);
        let p4 = R::wide_rescale(w);
        if p4 <= R::zero() {
            return None;
        }
        let l44 = R::sqrt(p4);
        let w = R::wide_add(R::wide_zero(), a.m54);
        let w = R::wide_sub_prod(w, l51, l41);
        let w = R::wide_sub_prod(w, l52, l42);
        let w = R::wide_sub_prod(w, l53, l43);
        let n54 = R::wide_rescale(w);
        let l54 = R::div(n54, l44);
        let w = R::wide_add(R::wide_zero(), a.m64);
        let w = R::wide_sub_prod(w, l61, l41);
        let w = R::wide_sub_prod(w, l62, l42);
        let w = R::wide_sub_prod(w, l63, l43);
        let n64 = R::wide_rescale(w);
        let l64 = R::div(n64, l44);
        let w = R::wide_add(R::wide_zero(), a.m55);
        let w = R::wide_sub_prod(w, l51, l51);
        let w = R::wide_sub_prod(w, l52, l52);
        let w = R::wide_sub_prod(w, l53, l53);
        let w = R::wide_sub_prod(w, l54, l54);
        let p5 = R::wide_rescale(w);
        if p5 <= R::zero() {
            return None;
        }
        let l55 = R::sqrt(p5);
        let w = R::wide_add(R::wide_zero(), a.m65);
        let w = R::wide_sub_prod(w, l61, l51);
        let w = R::wide_sub_prod(w, l62, l52);
        let w = R::wide_sub_prod(w, l63, l53);
        let w = R::wide_sub_prod(w, l64, l54);
        let n65 = R::wide_rescale(w);
        let l65 = R::div(n65, l55);
        let w = R::wide_add(R::wide_zero(), a.m66);
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

    /// `Cholesky6Trait::solve` before WP 11-OPT-2 (the same body, called, not inlined).
    fn solve_reference(self: Cholesky6<Fixed>, b: Vector6<Fixed>) -> Vector6<Fixed> {
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
        let w = R::wide_add(R::wide_zero(), b.a);
        let w = R::wide_sub_prod(w, self.l51, y1);
        let w = R::wide_sub_prod(w, self.l52, y2);
        let w = R::wide_sub_prod(w, self.l53, y3);
        let w = R::wide_sub_prod(w, self.l54, y4);
        let f5 = R::wide_rescale(w);
        let y5 = R::div(f5, self.l55);
        let w = R::wide_add(R::wide_zero(), b.b);
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
        Vector6 { x: x1, y: x2, z: x3, w: x4, a: x5, b: x6 }
    }

    fn fx(raw: i64) -> Fixed {
        Fixed { raw }
    }

    fn same(a: Cholesky6<Fixed>, b: Cholesky6<Fixed>) -> bool {
        a.l11 == b.l11
            && a.l21 == b.l21
            && a.l31 == b.l31
            && a.l41 == b.l41
            && a.l51 == b.l51
            && a.l61 == b.l61
            && a.l22 == b.l22
            && a.l32 == b.l32
            && a.l42 == b.l42
            && a.l52 == b.l52
            && a.l62 == b.l62
            && a.l33 == b.l33
            && a.l43 == b.l43
            && a.l53 == b.l53
            && a.l63 == b.l63
            && a.l44 == b.l44
            && a.l54 == b.l54
            && a.l64 == b.l64
            && a.l55 == b.l55
            && a.l65 == b.l65
            && a.l66 == b.l66
    }

    fn same_option(a: Option<Cholesky6<Fixed>>, b: Option<Cholesky6<Fixed>>) -> bool {
        match (a, b) {
            (Some(x), Some(y)) => same(x, y),
            (None, None) => true,
            _ => false,
        }
    }

    /// The symmetric matrix of diagonal `d` (6 raw values) and strict lower triangle `o` (15 raw
    /// values, row by row); the strictly upper triangle holds junk, which `new` must ignore.
    fn sym(d: Span<i64>, o: Span<i64>) -> Matrix6<Fixed> {
        Matrix6 {
            m11: fx(*d.at(0)),
            m12: fx(3),
            m13: fx(4),
            m14: fx(5),
            m15: fx(6),
            m16: fx(-6),
            m21: fx(*o.at(0)),
            m22: fx(*d.at(1)),
            m23: fx(-2),
            m24: fx(-1),
            m25: fx(0),
            m26: fx(1),
            m31: fx(*o.at(1)),
            m32: fx(*o.at(2)),
            m33: fx(*d.at(2)),
            m34: fx(6),
            m35: fx(-6),
            m36: fx(-5),
            m41: fx(*o.at(3)),
            m42: fx(*o.at(4)),
            m43: fx(*o.at(5)),
            m44: fx(*d.at(3)),
            m45: fx(1),
            m46: fx(2),
            m51: fx(*o.at(6)),
            m52: fx(*o.at(7)),
            m53: fx(*o.at(8)),
            m54: fx(*o.at(9)),
            m55: fx(*d.at(4)),
            m56: fx(-4),
            m61: fx(*o.at(10)),
            m62: fx(*o.at(11)),
            m63: fx(*o.at(12)),
            m64: fx(*o.at(13)),
            m65: fx(*o.at(14)),
            m66: fx(*d.at(5)),
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

    /// A raw value uniform in `[lo, lo + span]`.
    fn draw_pos(ref state: u128, lo: u128, span: u128) -> i64 {
        (lo + next(ref state) % (span + 1)).try_into().unwrap()
    }

    /// `n` raw values uniform in `[-bound, bound]`.
    fn draws(ref state: u128, n: u32, bound: u128) -> Span<i64> {
        let mut a = array![];
        for _ in 0..n {
            a.append(draw(ref state, bound));
        }
        a.span()
    }

    /// `n` raw values uniform in `[lo, lo + span]`.
    fn draws_pos(ref state: u128, n: u32, lo: u128, span: u128) -> Span<i64> {
        let mut a = array![];
        for _ in 0..n {
            a.append(draw_pos(ref state, lo, span));
        }
        a.span()
    }

    /// `new` against the reference, bit for bit: edge cases (zero, identity, 1-ulp pivots, a
    /// negative first and a zero later pivot, diagonals of the largest magnitude, the oracle input
    /// of the probe) and a deterministic sweep of diagonally dominant (positive-definite) and
    /// indefinite matrices of every magnitude.
    #[test]
    fn test_new_matches_reference() {
        let one: i64 = 0x100000000;
        let z15 = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span();
        let big: i64 = 0x7fffffffffffffff;
        let mut cases: Array<(Span<i64>, Span<i64>)> = array![
            (array![0, 0, 0, 0, 0, 0].span(), z15),
            (array![one, one, one, one, one, one].span(), z15),
            (array![1, 1, 1, 1, 1, 1].span(), z15),
            (array![-one, one, one, one, one, one].span(), z15),
            (
                array![one, one, one, one, one, one].span(),
                array![one, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span(),
            ),
            (array![big, big, big, big, big, big].span(), z15),
            (
                array![3654480394, 4075594332, 4932250777, 3475255779, 4388679014, 4212214819]
                    .span(),
                array![
                    90933453, -304406952, -1110029388, -302438381, 810457968, 743241296, -235454993,
                    432815, -713496402, 101179924, -273500147, 165140043, -664760589, 635633272,
                    -240687438,
                ]
                    .span(),
            ),
        ];
        let mut state: u128 = 0xc406;
        for k in 0..240_u32 {
            // Off-diagonal magnitudes from 2^-24 to 2^10; the diagonal dominates the 5 entries of
            // its row on 3 draws of 4 (positive definite), and is drawn on their scale otherwise.
            let ob: u128 = match k % 4 {
                0 => 0x100,
                1 => 0x100000000,
                2 => 0x10000000000,
                _ => 0x40000000000,
            };
            let o = draws(ref state, 15, ob);
            let d = if k % 4 == 3 {
                draws(ref state, 6, 2 * ob)
            } else {
                draws_pos(ref state, 6, 5 * ob + 1, 4 * ob)
            };
            cases.append((d, o));
        }
        let mut n = 0_u32;
        let mut factored = 0_u32;
        for c in cases.span() {
            let (d, o) = *c;
            let a = sym(d, o);
            let got = Cholesky6Trait::new(a);
            assert!(same_option(got, new_reference(a)));
            if got.is_some() {
                factored += 1;
            }
            n += 1;
        }
        assert!(n >= 200);
        assert!(factored >= 150);
    }

    /// `solve` against the reference, bit for bit: edge cases (identity factor, zero and 1-ulp
    /// right-hand sides, the probe's factor and right-hand side) and a deterministic sweep of
    /// factors (pivots in `[1/2, 4]`, entries in `[-1, 1]`) with right-hand sides from 2^-32 to
    /// 2^6.
    #[test]
    fn test_solve_matches_reference() {
        let u: u128 = 0x100000000;
        let one: i64 = 0x100000000;
        let mut cases: Array<(Cholesky6<Fixed>, Vector6<Fixed>)> = array![];
        let id = new_reference(
            sym(
                array![one, one, one, one, one, one].span(),
                array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].span(),
            ),
        )
            .unwrap();
        let f = new_reference(
            sym(
                array![3654480394, 4075594332, 4932250777, 3475255779, 4388679014, 4212214819]
                    .span(),
                array![
                    90933453, -304406952, -1110029388, -302438381, 810457968, 743241296, -235454993,
                    432815, -713496402, 101179924, -273500147, 165140043, -664760589, 635633272,
                    -240687438,
                ]
                    .span(),
            ),
        )
            .unwrap();
        let rhs = array![
            Vector6 { x: fx(0), y: fx(0), z: fx(0), w: fx(0), a: fx(0), b: fx(0) },
            Vector6 { x: fx(1), y: fx(-1), z: fx(1), w: fx(-1), a: fx(1), b: fx(-1) },
            Vector6 {
                x: fx(2829439348),
                y: fx(-7782455719),
                z: fx(7478168046),
                w: fx(-4127194961),
                a: fx(6320029132),
                b: fx(7729783779),
            },
        ];
        for b in rhs.span() {
            cases.append((id, *b));
            cases.append((f, *b));
        }
        let mut state: u128 = 0x5016;
        for k in 0..240_u32 {
            let f = Cholesky6 {
                l11: fx(draw_pos(ref state, u / 2, 7 * u / 2)),
                l21: fx(draw(ref state, u)),
                l31: fx(draw(ref state, u)),
                l41: fx(draw(ref state, u)),
                l51: fx(draw(ref state, u)),
                l61: fx(draw(ref state, u)),
                l22: fx(draw_pos(ref state, u / 2, 7 * u / 2)),
                l32: fx(draw(ref state, u)),
                l42: fx(draw(ref state, u)),
                l52: fx(draw(ref state, u)),
                l62: fx(draw(ref state, u)),
                l33: fx(draw_pos(ref state, u / 2, 7 * u / 2)),
                l43: fx(draw(ref state, u)),
                l53: fx(draw(ref state, u)),
                l63: fx(draw(ref state, u)),
                l44: fx(draw_pos(ref state, u / 2, 7 * u / 2)),
                l54: fx(draw(ref state, u)),
                l64: fx(draw(ref state, u)),
                l55: fx(draw_pos(ref state, u / 2, 7 * u / 2)),
                l65: fx(draw(ref state, u)),
                l66: fx(draw_pos(ref state, u / 2, 7 * u / 2)),
            };
            let bb: u128 = match k % 4 {
                0 => 0x100,
                1 => 0x100000000,
                2 => 0x1000000000,
                _ => 0x4000000000,
            };
            let v = Vector6 {
                x: fx(draw(ref state, bb)),
                y: fx(draw(ref state, bb)),
                z: fx(draw(ref state, bb)),
                w: fx(draw(ref state, bb)),
                a: fx(draw(ref state, bb)),
                b: fx(draw(ref state, bb)),
            };
            cases.append((f, v));
        }
        let mut n = 0_u32;
        for c in cases.span() {
            let (f, v) = *c;
            assert!(f.solve(v) == solve_reference(f, v));
            n += 1;
        }
        assert!(n >= 200);
    }
}
