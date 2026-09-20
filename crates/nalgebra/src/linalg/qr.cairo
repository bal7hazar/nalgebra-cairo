//! QR factorisation of the static square matrices (upstream `nalgebra::linalg::QR`), one unrolled
//! module per dimension: `Qr2`, `Qr3`, `Qr4`.
//!
//! `QrN::new(a)` always succeeds, like upstream, and yields `A = Q * R` with `Q` orthonormal and
//! `R` upper triangular with a **non-negative diagonal**. Both factors are stored in full: unlike
//! `LU`, whose two triangles pack into one matrix, `Q` is dense and has nothing to share with `R`,
//! so `q()` and `r()` are moves.
//!
//! **Sign convention.** Upstream stores the Householder reflectors and the signed diagonal of `R`,
//! but `QR::q()` and `QR::r()` hand the sign back: `r()` sets the diagonal to `diag.modulus()` and
//! `q()` folds `diag.signum()` into the reflection. The unpacked upstream factors therefore have
//! `r_ii >= 0`, which is checked on the 90 matrices of the `qr` oracle suite
//! (`tools/oracle/vectors/qr.json`: no negative diagonal entry at any size) and is exactly the
//! normalisation a Gram-Schmidt orthogonalisation produces. The factorisation of a non-singular
//! matrix is unique under it, so the oracle vectors are compared entry by entry, with no sign
//! flip — the `qr` README warns about `diag(r)` being negative, which describes the PACKED
//! `diag` field, not the unpacked `r()`.
//!
//! **Algorithm: modified Gram-Schmidt**, not Householder. Both were implemented and measured
//! against the oracle (see `qr3`, `test_householder_candidate_accuracy` and
//! `bench_qr3_new__alt_householder`). On the well-conditioned inputs the library targets, MGS is
//! **2.7x cheaper** (66 580 against 177 580 gas net) — it produces `Q` directly where Householder
//! normalises a reflector axis per step and then accumulates the reflections into `Q` — and it
//! lands **closer to upstream's own unpacked factors** (121 ulp against 402), because every extra
//! reflection is another rounding. Householder keeps a slightly tighter `QᵀQ = I` (29 ulp against
//! 36), which is not worth 111 000 gas here. The variant is kept as evidence (AGENTS.md rule 8).
//!
//! **Rank deficiency.** When a column of the working matrix collapses to exactly zero, `r_ii` is
//! zero and there is no direction to normalise. The column `q_i` is then set to **zero** rather
//! than completed to an orthonormal basis: `Q * R = A` still holds exactly (row `i` of `R` is
//! entirely zero, so `q_i` multiplies nothing), `is_invertible()` reports the deficiency, and
//! `solve` / `try_inverse` return `None`. `Q` is orthonormal if and only if `is_invertible()`.
//! Completing the basis would cost a `Vector3::orthonormal_basis` (two divisions) and a cross
//! product charged on EVERY call, since Sierra prices the worst-case branch: **+39 %** on `Qr3`
//! (92 370 against 66 580 gas net, `bench_qr3_new__alt_completed_basis`), and nothing in the
//! library consumes a rank-deficient `Q`. `Svd3`, whose `U` must be orthonormal for `recompose`
//! and `to_polar` to mean anything, does complete the basis — each decomposition decides whether
//! the property is worth the gas.
//!
//! Rejecting a singular matrix is not a saving here either: `solve` returning `None` costs what
//! the successful call costs (`bench_qrN_solve_singular__none`). Check `is_invertible` when the
//! answer changes what the caller does, not to save gas.

#[cfg(test)]
mod oracle_qr2;
#[cfg(test)]
mod oracle_qr3;
#[cfg(test)]
mod oracle_qr4;
pub mod qr2;
pub mod qr3;
pub mod qr4;

pub use qr2::{Matrix2QrTrait, Qr2, Qr2Trait};
pub use qr3::{Matrix3QrTrait, Qr3, Qr3Trait};
pub use qr4::{Matrix4QrTrait, Qr4, Qr4Trait};
