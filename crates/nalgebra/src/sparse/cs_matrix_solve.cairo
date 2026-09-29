//! The triangular solves of `CsMatrix` (upstream `sparse/cs_matrix_solve.rs`): `L x = b` and
//! `Lᵀ x = b` for a lower-triangular sparse `L` (its strict upper part is ignored, like
//! upstream's), dense or sparse `b`.
//!
//! Formulation: upstream solves column by column, scattering `b[i] -= x[j] * L[i, j]` into the
//! right-hand side; Cairo memory is write-once, so `L x = b` runs ROW by row on the transposed
//! pattern (`x[i] = (b[i] - sum_j L[i, j] x[j]) / L[i, i]`, the `x[j]` read back from the
//! solution built so far) and `Lᵀ x = b` column by column backwards (already a gather upstream).
//! Every `b[i] - sum` is ONE exact accumulation floored once, then divided (rounded to nearest):
//! two roundings per component, where upstream rounds every update.

pub use nalgebra_sparse::sparse::cs_matrix_solve::*;
