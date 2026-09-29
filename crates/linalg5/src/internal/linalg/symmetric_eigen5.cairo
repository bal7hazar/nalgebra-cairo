//! Internal, no stability promise: the crate-private items of `linalg::symmetric_eigen5` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// The 15 independent components of a symmetric 5x5 matrix, `mIJ` with `I <= J`
/// (crate-internal: the input of the Jacobi kernel and the Gram matrix of the SVD).
#[derive(Copy, Drop)]
pub struct Sym5<T> {
    pub m11: T,
    pub m12: T,
    pub m13: T,
    pub m14: T,
    pub m15: T,
    pub m22: T,
    pub m23: T,
    pub m24: T,
    pub m25: T,
    pub m33: T,
    pub m34: T,
    pub m35: T,
    pub m44: T,
    pub m45: T,
    pub m55: T,
}
