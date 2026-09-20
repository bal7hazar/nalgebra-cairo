//! Statically sized vectors and matrices (upstream `nalgebra::base`).

pub mod matrix2;
pub mod matrix3;
pub mod matrix4;
#[cfg(test)]
mod matrix_test_utils;
#[cfg(test)]
mod oracle_matrix2;
#[cfg(test)]
mod oracle_matrix2_inverse;
#[cfg(test)]
mod oracle_matrix3;
#[cfg(test)]
mod oracle_matrix3_inverse;
#[cfg(test)]
mod oracle_matrix4;
#[cfg(test)]
mod oracle_matrix4_inverse;
#[cfg(test)]
mod oracle_sym_matrix;
pub mod point2;
pub mod point3;
pub mod sym_matrix2;
pub mod sym_matrix3;
pub mod unit;
pub mod vector2;
pub mod vector3;
pub mod vector4;

pub use matrix2::Matrix2;
pub use matrix2::Matrix2Trait;
pub use matrix3::{Matrix3, Matrix3Trait};
pub use matrix4::{Matrix4, Matrix4Trait};
pub use point2::Point2;
pub use point3::Point3;
pub use sym_matrix2::{SymMatrix2, SymMatrix2Trait};
pub use sym_matrix3::{SymMatrix3, SymMatrix3Trait};
pub use unit::Unit;
pub use vector2::Vector2;
pub use vector3::Vector3;
pub use vector4::Vector4;
