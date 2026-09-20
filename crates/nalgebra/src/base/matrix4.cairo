//! `Matrix4`: a statically sized 4x4 matrix (upstream `nalgebra::Matrix4`).

/// A 4x4 matrix. `mRC` is the component at row `R`, column `C`.
///
/// Fields are declared in column-major order, so `Serde` matches upstream's storage order, while
/// `new` takes its arguments in row-major order like upstream.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Matrix4<T> {
    pub m11: T,
    pub m21: T,
    pub m31: T,
    pub m41: T,
    pub m12: T,
    pub m22: T,
    pub m32: T,
    pub m42: T,
    pub m13: T,
    pub m23: T,
    pub m33: T,
    pub m43: T,
    pub m14: T,
    pub m24: T,
    pub m34: T,
    pub m44: T,
}
