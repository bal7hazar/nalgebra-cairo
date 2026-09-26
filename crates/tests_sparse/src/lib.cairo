//! Package `nalgebra_tests_sparse` (WP 8.6-P20, P18): tests and gas benchmarks of the sparse
//! matrices (`nalgebra::sparse`: `CsMatrix`, the triangular solves, `CsCholesky`, `axpy_cs`,
//! `cumsum`), of the Matrix Market parser (`nalgebra::io`) and of the vector convolutions
//! (`Convolution`), through the public API.

#[cfg(test)]
mod benches;
#[cfg(test)]
mod cholesky;
#[cfg(test)]
mod convolution;
#[cfg(test)]
mod cs_matrix;
#[cfg(test)]
mod helpers;
#[cfg(test)]
mod matrix_market;
#[cfg(test)]
mod ops;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod oracle_sparse;
#[cfg(test)]
mod solve;
