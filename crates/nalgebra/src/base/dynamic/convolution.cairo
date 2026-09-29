//! Vector convolutions (upstream `linalg/convolution.rs`, `Vector::convolve_full` /
//! `convolve_same` / `convolve_valid`) on the static column vectors (`Matrix1`, `Vector2..6`)
//! and on `DVector`, behind the `dynamic` feature (their results are dynamic vectors).
//!
//! The kernel is reversed once, so every output component is the dot product of two contiguous
//! runs (`self[u0..=u1]` and the reversed kernel), ONE exact accumulation floored once
//! (`DynKernels::dot`); upstream accumulates rounded products. The kernel is any vector that
//! converts `Into<DVector>` (a `DVector`, a static column vector...), like upstream's generic
//! `Vector<T, D2, S2>`.

pub use nalgebra_dynamic::base::dynamic::convolution::*;
