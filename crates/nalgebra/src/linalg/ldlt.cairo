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
//! factorisation that consumes the leading principal minors in order matches the spatial-algebra
//! blocks of a `Matrix6` (its leading 3x3 block first).
//!
//! What is stored: the n(n-1)/2 strictly lower components of `L` as flat named fields (the unit
//! diagonal is implicit and never materialised) plus `D` as a `VectorN`, returned as such by `d()`.
//!
//! Numeric contract (AGENTS.md rule 4): every sum of products is accumulated EXACTLY in the
//! `Real::Wide` accumulator and floored ONCE; divisions are correctly rounded divisions, never a
//! multiplication by a rounded reciprocal (the `alt_recip` candidates of `benches.cairo` lose on
//! gas in `solve`; in `inverse` they are cheaper since `fixed` 0.3.0 but round twice per entry,
//! where upstream's substitution divides). `new` keeps the unrounded numerator of each
//! column as the column of `l·diag(d)` instead of recomputing `l_jk·d_k`: the `alt_products`
//! candidate that recomputes it is measurably dearer and less accurate, see `Ldlt2Trait::new`.
//!
//! NOT ported: upstream's `UDU` has no `solve` / `inverse` / `determinant` at all (it only exposes
//! `u` and `d`); those follow `Cholesky`'s surface here. `rank_one_update` is not ported either,
//! for the reason given in `cholesky.cairo`.

// the in-crate tests reach the internal items of the module (and the other modules' tests
// through `crate::linalg::...`) here (WP 9-NS9)

#[cfg(test)]
pub(crate) use nalgebra_linalg4::internal::linalg::ldlt::{
    Ldlt2, Ldlt2Trait, Ldlt3, Ldlt3Trait, Ldlt4, Ldlt4Trait,
};
#[cfg(test)]
pub(crate) use nalgebra_linalg6::internal::linalg::ldlt::{Ldlt6, Ldlt6Trait};

#[cfg(test)]
mod benches;

#[cfg(test)]
mod tests;
