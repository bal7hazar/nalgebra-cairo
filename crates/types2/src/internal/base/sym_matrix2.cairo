//! Internal, no stability promise: the crate-private items of `base::sym_matrix2` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// A symmetric 2x2 matrix `[[m11, m12], [m12, m22]]`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct SymMatrix2<T> {
    pub m11: T,
    pub m12: T,
    pub m22: T,
}
