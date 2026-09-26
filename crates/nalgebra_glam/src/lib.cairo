//! Conversions between nalgebra-cairo and glam-cairo types.
//!
//! The Cairo counterpart of nalgebra-rs's `convert-glam0XX` features
//! (`nalgebra/src/third_party/glam`):
//! a separate package, because Scarb has no optional dependencies and `glam` costs every
//! dependent of `nalgebra` compile time and memory (DESIGN D11). Modules mirror upstream's files:
//!
//! - `glam_matrix`: `Vector2..4`, `UnitVector2..4`, `Matrix2..4` <-> `Vec*` / `IVec*` / `UVec*` /
//!   `BVec*`, `Mat*`;
//! - `glam_point`, `glam_translation`: `Point2..4`, `Translation2..4` <-> the vectors;
//! - `glam_quaternion`, `glam_rotation`, `glam_unit_complex`: `Quaternion`, `UnitQuaternion`,
//!   `Rotation2..3`, `UnitComplex` <-> `Quat`, `Mat2`;
//! - `glam_isometry`, `glam_similarity`: `Isometry2..3`, `Similarity2..3` <-> tuples, vectors,
//!   quaternions and homogeneous matrices.
//!
//! Upstream's `From<X> for Y` is `Into<X, Y>` here, `TryFrom` is `TryInto` (returning an
//! `Option`). The impls are found by the compiler only when in scope: `use
//! nalgebra_glam::prelude::*;` imports all of them.
//!
//! The scalar is `fixed::Fixed` (glam-cairo's `Vec2` holds `Fixed`, not `f32`); glam-cairo has no
//! `f64` (`DVec*`, `DMat*`, `DQuat`), no aligned `Vec3A` and no integer vectors other than `IVec*`
//! (`i32`) and `UVec*` (`u32`): the impls of those upstream types are excluded (`interop`).

pub mod glam_isometry;
pub mod glam_matrix;
pub mod glam_point;
pub mod glam_quaternion;
pub mod glam_rotation;
pub mod glam_similarity;
pub mod glam_translation;
pub mod glam_unit_complex;
#[cfg(test)]
mod tests;

/// `use nalgebra_glam::prelude::*;`: every conversion impl.
pub mod prelude {
    pub use super::glam_isometry::*;
    pub use super::glam_matrix::*;
    pub use super::glam_point::*;
    pub use super::glam_quaternion::*;
    pub use super::glam_rotation::*;
    pub use super::glam_similarity::*;
    pub use super::glam_translation::*;
    pub use super::glam_unit_complex::*;
}
