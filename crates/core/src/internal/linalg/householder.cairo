//! Internal, no stability promise: the crate-private items of `linalg::householder` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// The Householder axis of one column vector shape (crate-private: the free function below is
/// upstream's interface).
pub trait HouseholderAxis<V, T> {
    fn reflection_axis_mut(ref column: V) -> (T, bool);
}
