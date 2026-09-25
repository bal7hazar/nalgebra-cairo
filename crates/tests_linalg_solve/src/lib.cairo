//! Package `nalgebra_tests_linalg_solve` (WP 8.5-P14a): tests and gas benchmarks of the triangular
//! solves (`MatrixSolve`, upstream `src/linalg/solve.rs`) on the VECTOR right-hand sides: the
//! kernels of every square, every one of the 26 forms, the zero-diagonal semantics. The matrix
//! right-hand sides are `nalgebra_tests_linalg_solve_rhs` (split for the compile budget).

#[cfg(test)]
mod oracle_solve;
#[cfg(test)]
mod solve;
