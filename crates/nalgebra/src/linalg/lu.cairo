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
#[cfg(test)]
mod oracle_matrix3_compare;

pub use lu2::{Lu2, Lu2Trait, Matrix2LuTrait};
pub use lu3::{Lu3, Lu3Trait, Matrix3LuTrait};
pub use lu4::{Lu4, Lu4Trait, Matrix4LuTrait};
pub use lu6::{Lu6, Lu6Trait, Matrix6LuTrait};

/// The row permutation of a 2x2 factorisation: the single transposition of step 1.
///
/// `p1` is the 1-based index of the row swapped with row 1 (`1` = no swap, `2` = rows 1 and 2
/// exchanged). Upstream: one entry of a `PermutationSequence`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Perm2 {
    /// Row swapped with row 1 at step 1, in `1..=2`.
    pub p1: u8,
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm2PartialEq of PartialEq<Perm2> {
    fn eq(lhs: @Perm2, rhs: @Perm2) -> bool {
        lhs.p1 == rhs.p1
    }
}

/// The row permutation of a 3x3 factorisation: the transpositions of steps 1 and 2.
///
/// `pK` is the 1-based index of the row swapped with row `K` at step `K` (`pK == K` = no swap),
/// so `P = T2 * T1` and the transpositions are replayed in the order `p1`, `p2`.
/// Upstream: a `PermutationSequence`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Perm3 {
    /// Row swapped with row 1 at step 1, in `1..=3`.
    pub p1: u8,
    /// Row swapped with row 2 at step 2, in `2..=3`.
    pub p2: u8,
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm3PartialEq of PartialEq<Perm3> {
    fn eq(lhs: @Perm3, rhs: @Perm3) -> bool {
        lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2
    }
}

/// The row permutation of a 4x4 factorisation: the transpositions of steps 1 to 3.
///
/// `pK` is the 1-based index of the row swapped with row `K` at step `K` (`pK == K` = no swap),
/// so `P = T3 * T2 * T1`. Upstream: a `PermutationSequence`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Perm4 {
    /// Row swapped with row 1 at step 1, in `1..=4`.
    pub p1: u8,
    /// Row swapped with row 2 at step 2, in `2..=4`.
    pub p2: u8,
    /// Row swapped with row 3 at step 3, in `3..=4`.
    pub p3: u8,
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
#[cfg(test)]
impl Perm4PartialEq of PartialEq<Perm4> {
    fn eq(lhs: @Perm4, rhs: @Perm4) -> bool {
        lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2 && lhs.p3 == rhs.p3
    }
}

/// The row permutation of a 6x6 factorisation: the transpositions of steps 1 to 5.
///
/// `pK` is the 1-based index of the row swapped with row `K` at step `K` (`pK == K` = no swap),
/// so `P = T5 * T4 * T3 * T2 * T1`. Upstream: a `PermutationSequence`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct Perm6 {
    /// Row swapped with row 1 at step 1, in `1..=6`.
    pub p1: u8,
    /// Row swapped with row 2 at step 2, in `2..=6`.
    pub p2: u8,
    /// Row swapped with row 3 at step 3, in `3..=6`.
    pub p3: u8,
    /// Row swapped with row 4 at step 4, in `4..=6`.
    pub p4: u8,
    /// Row swapped with row 5 at step 5, in `5..=6`.
    pub p5: u8,
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

/// Methods of `Perm2` (upstream `PermutationSequence<U2>`).
#[generate_trait]
pub impl Perm2Impl of Perm2Trait {
    /// The identity permutation of a 2x2 factorisation (no swap). Upstream:
    /// `PermutationSequence::identity`.
    #[inline(always)]
    fn identity() -> Perm2 {
        Perm2 { p1: 1 }
    }
}

/// Methods of `Perm3` (upstream `PermutationSequence<U3>`).
#[generate_trait]
pub impl Perm3Impl of Perm3Trait {
    /// The identity permutation of a 3x3 factorisation (no swap). Upstream:
    /// `PermutationSequence::identity`.
    #[inline(always)]
    fn identity() -> Perm3 {
        Perm3 { p1: 1, p2: 2 }
    }
}

/// Methods of `Perm4` (upstream `PermutationSequence<U4>`).
#[generate_trait]
pub impl Perm4Impl of Perm4Trait {
    /// The identity permutation of a 4x4 factorisation (no swap). Upstream:
    /// `PermutationSequence::identity`.
    #[inline(always)]
    fn identity() -> Perm4 {
        Perm4 { p1: 1, p2: 2, p3: 3 }
    }
}

/// Methods of `Perm6` (upstream `PermutationSequence<U6>`).
#[generate_trait]
pub impl Perm6Impl of Perm6Trait {
    /// The identity permutation of a 6x6 factorisation (no swap). Upstream:
    /// `PermutationSequence::identity`.
    #[inline(always)]
    fn identity() -> Perm6 {
        Perm6 { p1: 1, p2: 2, p3: 3, p4: 4, p5: 5 }
    }
}
