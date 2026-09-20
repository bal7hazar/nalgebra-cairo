//! `Vector2`: a statically sized 2-dimensional column vector (upstream `nalgebra::Vector2`).

/// A 2-dimensional column vector.
#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}
