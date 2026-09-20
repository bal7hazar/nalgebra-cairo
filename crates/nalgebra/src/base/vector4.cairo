//! `Vector4`: a statically sized 4-dimensional column vector (upstream `nalgebra::Vector4`).

/// A 4-dimensional column vector.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Vector4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}
