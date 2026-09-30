//! LU factorisation with partial pivoting of the static square matrices (upstream
//! `nalgebra::linalg::lu`), one unrolled module per dimension: `Lu2`, `Lu3`, `Lu4`, `Lu6`.
//!
//! `LuN::new(a)` always succeeds, like upstream, and yields `P * A = L * U`: `L` unit lower
//! triangular, `U` upper triangular (both packed in one `MatrixN`, upstream's storage) and `P`
//! the row permutation, here the compact `PermN` defined below instead of upstream's
//! heap-allocated `PermutationSequence`. `solve`, `try_inverse` and `determinant` are then
//! `Option`-returning or total methods on the factorisation; `MatrixNLuTrait::lu` is the
//! upstream-named entry point (`matrix.lu()`), and `Matrix6LuTrait` additionally carries
//! `determinant`, `try_inverse` and `solve` for 6x6 matrices, which have no closed form worth
//! writing.
//!
//! Rejecting a singular matrix is not a saving: Cairo charges the worst-case path of a function,
//! so `solve` / `try_inverse` returning `None` costs what the successful call costs (measured:
//! `bench_luN_solve_singular__none` against `bench_luN_solve__substitution`, exactly as
//! `bench_matrix3_try_inverse_singular__none` already shows for the closed form). Check
//! `is_invertible` when the answer changes what the caller does, not to save gas.
//!
//! A `PermN` stores the transposition chosen at each of the `N - 1` elimination steps: `pK` is
//! the 1-based index of the row swapped with row `K` at step `K`, so `pK == K` means "no swap".
//! That is upstream's representation minus the allocation, it is `Copy`, and applying it (or its
//! inverse) is a fixed chain of comparisons and moves — the crate-internal `LuN::permute` /
//! `LuN::permute_rows` and the column permutation inside `LuN::try_inverse`. Building the `N x N`
//! permutation matrix is deliberately not offered: nothing in the library needs it, and it would
//! cost a full matrix product to use.

pub use lu2::{Lu2, Lu2Trait, Matrix2LuTrait};
pub use lu3::{Lu3, Lu3Trait, Matrix3LuTrait};
pub use lu4::{Lu4, Lu4Trait, Matrix4LuTrait};
pub use lu6::{Lu6, Lu6Trait, Matrix6LuTrait};
pub use perm1_5::{Perm1, Perm1Trait, Perm5, Perm5Trait};
pub mod lu2;
pub mod lu3;
pub mod lu4;
pub mod lu6;
#[cfg(test)]
mod oracle_lu2;
#[cfg(test)]
mod oracle_lu3;
#[cfg(test)]
mod oracle_lu4;
#[cfg(test)]
mod oracle_lu6;
pub mod perm1_5;

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm2PartialEq of PartialEq<Perm2> {
    fn eq(lhs: @Perm2, rhs: @Perm2) -> bool {
        lhs.p1 == rhs.p1
    }
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm3PartialEq of PartialEq<Perm3> {
    fn eq(lhs: @Perm3, rhs: @Perm3) -> bool {
        lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2
    }
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm4PartialEq of PartialEq<Perm4> {
    fn eq(lhs: @Perm4, rhs: @Perm4) -> bool {
        lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2 && lhs.p3 == rhs.p3
    }
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm6PartialEq of PartialEq<Perm6> {
    fn eq(lhs: @Perm6, rhs: @Perm6) -> bool {
        lhs.p1 == rhs.p1
            && lhs.p2 == rhs.p2
            && lhs.p3 == rhs.p3
            && lhs.p4 == rhs.p4
            && lhs.p5 == rhs.p5
    }
}
pub use nalgebra_static6_wide::linalg::lu::{Perm6, Perm6Impl, Perm6Trait};
pub use nalgebra_types2::linalg::lu::{Perm2, Perm2Impl, Perm2Trait};
pub use nalgebra_types3::linalg::lu::{Perm3, Perm3Impl, Perm3Trait};
pub use nalgebra_types4::linalg::lu::{Perm4, Perm4Impl, Perm4Trait};
