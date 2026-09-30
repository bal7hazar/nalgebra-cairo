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

pub use nalgebra_linalg2::linalg::cholesky::*;
pub use nalgebra_linalg3::linalg::cholesky::*;
pub use nalgebra_linalg4::linalg::cholesky::*;
pub use nalgebra_linalg6::linalg::cholesky::*;

/// Test-only field-wise equality (upstream `Cholesky2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Cholesky2PartialEq<T, +PartialEq<T>> of PartialEq<Cholesky2<T>> {
    fn eq(lhs: @Cholesky2<T>, rhs: @Cholesky2<T>) -> bool {
        lhs.l11 == rhs.l11 && lhs.l21 == rhs.l21 && lhs.l22 == rhs.l22
    }
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
