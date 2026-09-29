//! Internal, no stability promise: the crate-private items of `linalg::householder_steps` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// Run-time column-major access to a static shape: the interface of the building blocks below
/// (their indices are run-time values, like upstream's). Crate-private: the free functions are
/// upstream's interface.
pub trait ColumnMajor<M, T> {
    /// The number of rows.
    fn nrows() -> usize;
    /// The number of columns.
    fn ncols() -> usize;
    /// The components in column-major order.
    fn to_column_major(self: M) -> Array<T>;
    /// The matrix of the column-major components `data` (panics with `nalgebra: wrong slice
    /// length` unless there are exactly `nrows * ncols`).
    fn from_column_major(data: Span<T>) -> M;
}
