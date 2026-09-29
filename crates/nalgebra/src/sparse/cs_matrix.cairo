//! `CsMatrix<T>`: a compressed sparse column matrix (upstream's legacy
//! `nalgebra::sparse::CsMatrix`, `CsMatrix<T, Dyn, Dyn, CsVecStorage<T, Dyn, Dyn>>`), its methods,
//! operators and conversions.
//!
//! Storage (upstream's `CsVecStorage`): `nrows`, `ncols`, the column pointers `p`, the row
//! indices `i` and the values `vals`, the entries of column `j` being `i[p[j]..p[j + 1]]` /
//! `vals[p[j]..p[j + 1]]`. Deviation: `p` keeps the `ncols + 1` pointers of the usual CSC layout
//! (upstream stores `ncols` of them and ends the last column at `len()`); the `p()` accessor
//! returns upstream's `ncols` first ones. Every constructor (`from_triplet`, `From<Matrix>`, the
//! operations) produces SORTED, deduplicated columns, the invariant the kernels rely on (as
//! upstream's). The spans are views of write-once arrays: `CsMatrix` is `Copy` (upstream
//! `Clone`) and every operation builds new spans (`sparse/cs_utils.cairo`: merges of sorted runs
//! and gathers stand for upstream's scatters).

pub use nalgebra_sparse::sparse::cs_matrix::*;
