//! `Vector3`: a statically sized 3-dimensional column vector (upstream `nalgebra::Vector3`).

/// A 3-dimensional column vector.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}
