//! Internal, no stability promise: the crate-private items of `linalg::lu_steps` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// The elimination steps of one square shape (crate-private: the free functions below are
/// upstream's interface).
pub trait LuSteps<M, T> {
    fn gauss_step(ref matrix: M, diag: T, i: usize);
    fn gauss_step_swap(ref matrix: M, diag: T, i: usize, piv: usize);
}
