//! `Orthographic3`: a 3D orthographic projection stored as its homogeneous `Matrix4` (upstream
//! `nalgebra::Orthographic3`), WP 8.4-P11b.
//!
//! The matrix has upstream's OpenGL layout: `m11 = 2 / (right - left)`, `m14 = -(right + left) /
//! (right - left)`, `m22 = 2 / (top - bottom)`, `m24 = -(top + bottom) / (top - bottom)`, `m33 =
//! -2 / (zfar - znear)`, `m34 = -(zfar + znear) / (zfar - znear)`, `m44 = 1`, every other entry
//! zero. Like upstream, the accessors, the setters and the projections read and write those seven
//! entries only; a matrix given to `from_matrix_unchecked` is trusted to have that structure.
//!
//! API split (the pattern of `UnitQuaternionTrait` / `UnitQuaternionAngleTrait`):
//! - `Orthographic3Trait` (`Real` scalar): everything but `from_fov`;
//! - `Orthographic3AngleTrait` (`Real` + `Transcendental`): `from_fov`;
//! - `Matrix4OrthographicTrait`: upstream's `Matrix4::new_orthographic` (`base/cg.rs`), which is
//!   `Orthographic3::new(..).into_inner()`;
//! - `Into<Orthographic3, Matrix4>`: upstream's `From<Orthographic3> for Matrix4`.
//!
//! Numeric contract (AGENTS.md): every division is ONE correctly rounded `Real::div` (to nearest,
//! ties to even, like `f64 /`), every product one floored fixed-point product, every product-sum
//! one fused kernel (`Real::mul_add`); overflow and division by zero panic (upstream's `inf` /
//! `NaN`). Upstream's `relative_eq!(a, b)` assertions use the default tolerances of DESIGN D3
//! (`default_epsilon` = 1 ulp, absolute and relative).

pub use nalgebra_geometry4::geometry::orthographic3::*;
