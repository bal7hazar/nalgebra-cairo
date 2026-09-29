//! `Vector::axpy_cs` (upstream `sparse/cs_matrix_ops.rs`): `y = alpha * x + beta * y` for a dense
//! vector `y` and a sparse one-column `x` (a `CsVector`), on the static column vectors
//! (`Matrix1`, `Vector2..6`) and on `DVector`.

pub use nalgebra_sparse::sparse::cs_matrix_ops::*;
