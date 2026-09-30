//! Internal, no stability promise: the crate-private items of `linalg::balancing` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// The balancing of one square shape `M` with its diagonal vector `V` (crate-private: the free
/// functions below are upstream's interface).
pub trait Balancing<M, V> {
    fn balance_parlett_reinsch(ref matrix: M) -> V;
    fn unbalance(ref m: M, d: V);
}
