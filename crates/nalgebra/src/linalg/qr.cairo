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
pub mod qr1;
pub mod qr1x2;
pub mod qr1x3;
pub mod qr1x4;
pub mod qr1x5;
pub mod qr1x6;
pub mod qr2;
pub mod qr2x1;
pub mod qr2x3;
pub mod qr2x4;
pub mod qr2x5;
pub mod qr2x6;
pub mod qr3;
pub mod qr3x1;
pub mod qr3x2;
pub mod qr3x4;
pub mod qr3x5;
pub mod qr3x6;
pub mod qr4;
pub mod qr4x1;
pub mod qr4x2;
pub mod qr4x3;
pub mod qr4x5;
pub mod qr4x6;
pub mod qr5;
pub mod qr5x1;
pub mod qr5x2;
pub mod qr5x3;
pub mod qr5x4;
pub mod qr5x6;
pub mod qr6;
pub mod qr6x1;
pub mod qr6x2;
pub mod qr6x3;
pub mod qr6x4;
pub mod qr6x5;
pub use qr1::{Matrix1QrTrait, Qr1, Qr1Trait};
pub use qr1x2::{Qr1x2, Qr1x2Trait, RowVector2QrTrait};
pub use qr1x3::{Qr1x3, Qr1x3Trait, RowVector3QrTrait};
pub use qr1x4::{Qr1x4, Qr1x4Trait, RowVector4QrTrait};
pub use qr1x5::{Qr1x5, Qr1x5Trait, RowVector5QrTrait};
pub use qr1x6::{Qr1x6, Qr1x6Trait, RowVector6QrTrait};

pub use qr2::{Matrix2QrTrait, Qr2, Qr2Trait};
pub use qr2x1::{Qr2x1, Qr2x1Trait, Vector2QrTrait};
pub use qr2x3::{Matrix2x3QrTrait, Qr2x3, Qr2x3Trait};
pub use qr2x4::{Matrix2x4QrTrait, Qr2x4, Qr2x4Trait};
pub use qr2x5::{Matrix2x5QrTrait, Qr2x5, Qr2x5Trait};
pub use qr2x6::{Matrix2x6QrTrait, Qr2x6, Qr2x6Trait};
pub use qr3::{Matrix3QrTrait, Qr3, Qr3Trait};
pub use qr3x1::{Qr3x1, Qr3x1Trait, Vector3QrTrait};
pub use qr3x2::{Matrix3x2QrTrait, Qr3x2, Qr3x2Trait};
pub use qr3x4::{Matrix3x4QrTrait, Qr3x4, Qr3x4Trait};
pub use qr3x5::{Matrix3x5QrTrait, Qr3x5, Qr3x5Trait};
pub use qr3x6::{Matrix3x6QrTrait, Qr3x6, Qr3x6Trait};
pub use qr4::{Matrix4QrTrait, Qr4, Qr4Trait};
pub use qr4x1::{Qr4x1, Qr4x1Trait, Vector4QrTrait};
pub use qr4x2::{Matrix4x2QrTrait, Qr4x2, Qr4x2Trait};
pub use qr4x3::{Matrix4x3QrTrait, Qr4x3, Qr4x3Trait};
pub use qr4x5::{Matrix4x5QrTrait, Qr4x5, Qr4x5Trait};
pub use qr4x6::{Matrix4x6QrTrait, Qr4x6, Qr4x6Trait};
pub use qr5::{Matrix5QrTrait, Qr5, Qr5Trait};
pub use qr5x1::{Qr5x1, Qr5x1Trait, Vector5QrTrait};
pub use qr5x2::{Matrix5x2QrTrait, Qr5x2, Qr5x2Trait};
pub use qr5x3::{Matrix5x3QrTrait, Qr5x3, Qr5x3Trait};
pub use qr5x4::{Matrix5x4QrTrait, Qr5x4, Qr5x4Trait};
pub use qr5x6::{Matrix5x6QrTrait, Qr5x6, Qr5x6Trait};
pub use qr6::{Matrix6QrTrait, Qr6, Qr6Trait};
pub use qr6x1::{Qr6x1, Qr6x1Trait, Vector6QrTrait};
pub use qr6x2::{Matrix6x2QrTrait, Qr6x2, Qr6x2Trait};
pub use qr6x3::{Matrix6x3QrTrait, Qr6x3, Qr6x3Trait};
pub use qr6x4::{Matrix6x4QrTrait, Qr6x4, Qr6x4Trait};
pub use qr6x5::{Matrix6x5QrTrait, Qr6x5, Qr6x5Trait};
