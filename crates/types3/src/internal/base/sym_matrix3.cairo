//! Internal, no stability promise: the crate-private items of `base::sym_matrix3` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// A symmetric 3x3 matrix `[[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]]`.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct SymMatrix3<T> {
    pub m11: T,
    pub m12: T,
    pub m13: T,
    pub m22: T,
    pub m23: T,
    pub m33: T,
}
