//! Points (upstream `nalgebra::Point`): the panic messages shared by every point type. The
//! completion of `Point2` / `Point3` (`Point2ExtTrait` / `Point3ExtTrait`) lives in the facade
//! `nalgebra` (`geometry::point`); `Point2Index` / `Point2PartialOrd` (and the `Point3` ones) in
//! the modules of their types (`base::point2` / `base::point3`), where Cairo finds them.

/// Panic messages of the point types (stable API).
pub mod errors {
    /// `p[i]` with `i` at least the dimension.
    pub const INDEX_OUT_OF_BOUNDS: felt252 = 'nalgebra: index out of bounds';
    /// `from_slice` of a span whose length is not the dimension.
    pub const WRONG_SLICE_LENGTH: felt252 = 'nalgebra: wrong slice length';
}
