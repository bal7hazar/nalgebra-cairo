// Specialisations of `Matrix6`, spliced verbatim by shapegen.py (format: `library.py`). The
// kernels are the templates' (6-term `Real::Wide` chains); these sections keep the measured notes
// of the hand-written block type (the block alternatives stay as benchmarks) in the doc comments.
// @doc module
//!
//! The 6-term rows of `*`, `mul_mat` and `tr_mul` (by a `Matrix6` or a `Vector6`) use the
//! `Real::Wide` accumulator, so a product is the exact 6x6 product with one rescale per output
//! scalar — NOT the sum of rounded 3x3 block products (which rounds twice; kept as
//! `bench_matrix6_mul__alt_blocks` with the test showing the drift).
//!
//! The determinant, the inverse and the decompositions of a 6x6 are NOT here: they are the LU /
//! Cholesky kernels of `linalg` (DESIGN D6). No block formula gives them cheaply — `try_inverse`
//! through the Schur complement needs two 3x3 inversions and four 3x3 products, each rounding
//! again, and is numerically far worse than a pivoted factorisation.
// @doc mul_mat Vector6
/// `self * v` (`Matrix6 * Vector6`): each of the six output components is the exact sum of the
/// six products of a row by `v`, accumulated in `Real::Wide` and rescaled ONCE. Panics on overflow
/// of a component. Upstream: `self * v`.
///
/// Summing the rounded products of the 3x3 blocks by the 3-vector halves of `v` rounds twice AND
/// costs 1.67x more gas (`bench_matrix6_mul_vec__alt_blocks`: 37 770 against 22 560 net).
// @doc tr_mul Vector6
/// `selfᵀ * v`, a `Vector6`: `mul_mat` of the transposed components, so one 6-term `Real::Wide`
/// accumulation and ONE rescale per component. Panics on overflow. Upstream: `self.tr_mul(&v)`.
///
/// Exactly as expensive as a kernel that reads `self` column by column (measured:
/// `bench_matrix6_tr_mul_vec__fused` and `__alt_transpose_then_mul_vec` are both 21 910 net):
/// `transpose` only relabels SSA values, so it is free.
// @doc tr_mul Matrix6
/// `selfᵀ * rhs`, a `Matrix6`: `mul_mat` (`*`) of the transposed components, so 36 six-term
/// `Real::Wide` accumulations and one rescale per output component. Panics on overflow. Upstream:
/// `tr_mul`.
///
/// Exactly as expensive as a kernel that reads `self` column by column (measured:
/// `bench_matrix6_tr_mul__fused` and `__alt_transpose_then_mul` are both 108 910 net), for the
/// same reason as `tr_mul` by a `Vector6`: transposing is free.
// @doc Mul
/// `a * b` (matrix product): 36 six-term `Real::Wide` accumulations, ONE rounding and one overflow
/// check per output scalar. Panics on overflow.
///
/// Deliberately NOT the block composition `m11 * n11 + m12 * n21` of the 3x3 blocks, which rounds
/// each 3x3 product before adding (two roundings per output scalar, a different result, and 2.00x
/// the gas: 229 850 against 114 660 net, 72 fused kernels plus 36 additions instead of 36); that
/// candidate is kept as `bench_matrix6_mul__alt_blocks`.
// @doc is_identity
/// Whether every component is within `ulps` smallest units of the identity's.
/// Upstream: `is_identity(eps)`, with the tolerance in raw units instead of a float epsilon.
///
/// One `#[inline(always)]` 36-term `&&` chain. Measured (raw gas, identity / first component
/// off): 70 630 / 25 740, against 74 340 / 36 090 for the four 3x3 blocks through the non-inlined
/// `Matrix3` kernels (the former block layout) and 92 260 / 92 460 for the same chain as a call
/// (`bench_matrix6_is_identity__alt_blocks*`, `__alt_not_inlined*`).
// @doc abs_diff_eq
/// Whether every component of `self` is within `ulps` smallest units of `other`'s.
/// Upstream: `abs_diff_eq`, with the tolerance in raw units instead of a float epsilon.
///
/// One `#[inline(always)]` 36-term `&&` chain, like `is_identity`. Measured (raw gas, all compared
/// / first component differs): 78 030 / 31 690, against 85 340 / 44 390 for the four 3x3 blocks
/// through the non-inlined `Matrix3` kernel and 103 260 / 103 460 for the same chain as a call
/// (`bench_matrix6_abs_diff_eq__alt_blocks*`, `__alt_not_inlined*`).
