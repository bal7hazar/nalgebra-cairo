//! Package `nalgebra_tests_linalg_factor` (WP 8.5-P14a): tests and gas benchmarks of the
//! Givens rotations, the permutation sequences, upstream's LU / Householder building blocks
//! (`gauss_step`, `gauss_step_swap`, `try_invert_to`, `reflection_axis_mut`) and the LU / QR /
//! Cholesky / UDU API completion, through the public API.

#[cfg(test)]
mod decompositions;
#[cfg(test)]
mod givens;
#[cfg(test)]
mod permutation;
#[cfg(test)]
mod steps;
