//! `Lu6`: the LU factorisation with partial pivoting of a `Matrix6` (upstream
//! `nalgebra::linalg::LU` on a 6x6 matrix).
//!
//! `P * A = L * U`, with `L` unit lower triangular, `U` upper triangular and `P` the product of the
//! 5 row transpositions chosen by partial pivoting. Both factors share one `Matrix6` like upstream
//! (strict lower triangle = `L`, whose unit diagonal is implicit; upper triangle = `U`) and the
//! permutation is the compact `Perm6`.
//!
//! Everything is unrolled (DESIGN D4: no loop in static code) and every sum of products goes
//! through a fused `Real` kernel — `mul_add` for a single product, the explicit `Real::Wide`
//! accumulator beyond that — so each output scalar is floored once and range-checked once
//! (AGENTS.md rule 4).
//!
//! Partial pivoting costs 15 `abs` and comparisons plus 5 conditional row swaps (moves only), and
//! every swap duplicates the row it moves, so it also costs Sierra statements. Dropping it would be
//! cheaper and shorter and it is not an option: without it the pivot of step `k` is whatever sits
//! at `a_kk`, nothing bounds `|l_ik|`, and a matrix as ordinary as a permuted identity factors with
//! a zero pivot. `bench_lu6_new__alt_no_pivot` and `test_no_pivot_candidate_is_wrong` keep the
//! measurement and the counter-example. Upstream has no unpivoted variant either, only `LU`
//! (partial pivoting) and `FullPivLU` (complete pivoting).

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;
// the crate-private kernels the in-crate tests use (`nalgebra_static6_wide::internal`)
#[cfg(test)]
use nalgebra_static6_wide::internal::linalg::lu::lu6::Lu6InternalTrait;


// the test-only `PartialEq` of `Perm6` (`linalg::lu`, not `Perm6`'s module since the split)
#[cfg(test)]
use super::Perm6PartialEq;

/// Test-only field-wise equality (upstream `Lu6` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
#[cfg(test)]
impl Lu6PartialEq<T, +PartialEq<T>> of PartialEq<Lu6<T>> {
    fn eq(lhs: @Lu6<T>, rhs: @Lu6<T>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}
pub use nalgebra_static6_wide::linalg::lu::lu6::*;
