//! `AbstractRotation`: the operations every rotation type provides (upstream
//! `nalgebra::AbstractRotation`, `geometry/abstract_rotation.rs`), implemented by `Rotation2`,
//! `Rotation3`, `UnitComplex` and `UnitQuaternion` like upstream.
//!
//! Upstream's `AbstractRotation<T, D>` is generic over the dimension and names the vector and
//! point types through it; the Cairo trait is generic over the rotation type `R` and names them
//! with the associated types `Vector` and `Point` (`Vector2` / `Point2` for the 2D rotations,
//! `Vector3` / `Point3` for the 3D ones). Every method forwards to the rotation's own method of the
//! same name (no arithmetic here): the costs and roundings are those of `Rotation2Trait` & co.
//!
//! The methods have the names of the inherent methods: with both `AbstractRotation` and, say,
//! `Rotation3Trait` imported, `r.inverse()` is ambiguous — import the trait alone in generic
//! code, or call it by path (`AbstractRotation::inverse(r)`).
pub use nalgebra_geometry3::geometry::abstract_rotation::*;
