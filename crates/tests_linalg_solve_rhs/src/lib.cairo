//! Package `nalgebra_tests_linalg_solve_rhs` (WP 8.5-P14a): tests and gas benchmarks of the
//! triangular solves (`MatrixSolve`) on the MATRIX right-hand sides: the three kernels of every
//! (square, `R x C` right-hand side, `C >= 2`) pair, and the per-column alternative that lost.

#[cfg(test)]
mod solve;
