//! Internal, no stability promise: the crate-private items of `base::matrix_view` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// The length of a row vector `S` (`RowPart`).
pub trait RowVectorLen<S> {
    fn len() -> usize;
}

/// The length of a column vector `S` (`ColumnPart`).
pub trait ColumnVectorLen<S> {
    fn len() -> usize;
}
