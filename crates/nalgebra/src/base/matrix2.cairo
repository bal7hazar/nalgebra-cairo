//! `Matrix2`: a statically sized 2x2 matrix (upstream `nalgebra::Matrix2`).

/// A 2x2 matrix. `mRC` is the component at row `R`, column `C`.
///
/// Fields are declared in column-major order, so `Serde` matches upstream's storage order, while
/// `new` takes its arguments in row-major order like upstream.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Matrix2<T> {
    pub m11: T,
    pub m21: T,
    pub m12: T,
    pub m22: T,
}
