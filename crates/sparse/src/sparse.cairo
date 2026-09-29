//! Sparse matrices: upstream's legacy `nalgebra::sparse` (`CsMatrix`, `CsVector`, `CsCholesky`,
//! the sparse triangular solves, `Vector::axpy_cs`, `cumsum`). Behind the Scarb feature `sparse`
//! (in `default`, DESIGN D9; it enables `dynamic`: the dense counterparts are `DMatrix` /
//! `DVector`).
//!
//! - `cs_matrix`: the compressed sparse column type, its methods, operators and conversions;
//! - `cs_matrix_solve`: the triangular solves (dense and sparse right-hand sides);
//! - `cs_matrix_cholesky`: the sparse Cholesky factorization (symbolic analysis, then numeric);
//! - `cs_matrix_ops`: `axpy_cs` on the dense vectors;
//! - `cs_utils`: `cumsum` and the index kernels (key sorts, pattern transposition, gathers).
//!
//! Cairo memory is write-once: the kernels replace upstream's scatters into work arrays by
//! sorts of packed keys, gathers and row-oriented (gather) formulations of the solves and of the
//! factorization; the symbolic analysis keeps upstream's elimination tree with `Felt252Dict`
//! work arrays. Every sum of products is ONE exact accumulation floored once (`Real::Wide`).

mod cs_kernels;
pub mod cs_matrix;
pub mod cs_matrix_cholesky;
pub mod cs_matrix_ops;
pub mod cs_matrix_solve;
pub mod cs_utils;
pub mod errors;

pub use cs_matrix::{CsMatrix, CsMatrixTrait, CsVector};
pub use cs_matrix_cholesky::{CsCholesky, CsCholeskyTrait};
pub use cs_matrix_ops::AxpyCs;
pub use cs_matrix_solve::CsMatrixSolveTrait;

pub use cs_utils::cumsum;
