//! Internal, no stability promise: the crate-private items of `linalg::symmetric_eigen4` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// The 10 independent components of a symmetric 4x4 matrix, `mIJ` with `I <= J`
/// (crate-internal: the input of the Jacobi kernel and the Gram matrix of the SVD).
#[derive(Copy, Drop)]
pub struct Sym4<T> {
    pub m11: T,
    pub m12: T,
    pub m13: T,
    pub m14: T,
    pub m22: T,
    pub m23: T,
    pub m24: T,
    pub m33: T,
    pub m34: T,
    pub m44: T,
}
