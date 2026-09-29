//! `CsCholesky<T>`: the Cholesky factorization `A = L Lᵀ` of a sparse symmetric positive-definite
//! matrix (upstream's legacy `nalgebra::sparse::CsCholesky`): a symbolic analysis of the
//! pattern (`new_symbolic`), then numeric factorizations of values with that pattern
//! (`decompose_left_looking` / `decompose_up_looking`), `L` read with `l` / `unwrap_l`.
//!
//! Symbolic analysis (upstream's `elimination_tree` / `reach` / `nonzero_pattern`): the pattern
//! of row `k` of `L` is the set of nodes reached from the entries `A[i, k]`, `i < k`, by walking
//! up the elimination tree until a node already reached for `k`; the tree is built during the
//! same pass (a node's parent is the first row that reaches it, Liu's row-subtree
//! characterisation), the `parent` and `mark` work arrays are `Felt252Dict`s (Cairo memory is
//! write-once; they replace upstream's `forest` / `ancestor` / `marks` vectors). Every walk goes
//! up the tree, so each row's pattern is a union of ascending chains: kept sorted by merges. The
//! pattern of `L` is upstream's (the column counts are its column lengths), with the diagonal
//! always stored.
//!
//! Numeric factorization: row by row (up-looking): `L[k, j] = (A[k, j] - sum_{c < j} L[k, c]
//! L[j, c]) / L[j, j]` over the pattern of row `k` in ascending `j`, then `L[k, k] = sqrt(A[k, k]
//! - sum_c L[k, c]^2)`, each sum ONE exact accumulation floored once (the sparse dot products of
//! two sorted rows by merge), the quotient rounded to nearest. Both entry points run this gather
//! kernel (upstream's left-looking variant scatters column updates into a work vector, which
//! write-once memory cannot do); as upstream, `decompose_left_looking` reads the LOWER triangle
//! of the values (`A[i, k]`, `i >= k`) and `decompose_up_looking` the UPPER one (`A[i, k]`,
//! `i <= k`): identical results on symmetric input (`new` reads the upper one, see its doc).
//! `L` is then gathered into its column layout.

pub use nalgebra_sparse::sparse::cs_matrix_cholesky::*;
