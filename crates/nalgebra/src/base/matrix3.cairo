//! `Matrix3`: a statically sized 3x3 matrix (upstream `nalgebra::Matrix3`).

/// A 3x3 matrix. `mRC` is the component at row `R`, column `C`.
///
/// Fields are declared in column-major order, so `Serde` matches upstream's storage order, while
/// `new` takes its arguments in row-major order like upstream.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Matrix3<T> {
    pub m11: T,
    pub m21: T,
    pub m31: T,
    pub m12: T,
    pub m22: T,
    pub m32: T,
    pub m13: T,
    pub m23: T,
    pub m33: T,
}
