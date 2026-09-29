//! `Perspective3`: a 3D perspective projection stored as its homogeneous `Matrix4` (upstream
//! `nalgebra::Perspective3`), WP 8.4-P11b.
//!
//! The matrix has upstream's OpenGL layout: `m11 = m22 / aspect`, `m22 = 1 / tan(fovy / 2)`,
//! `m33 = (zfar + znear) / (znear - zfar)`, `m34 = 2 · zfar · znear / (znear - zfar)`,
//! `m43 = -1`, every other entry zero. Like upstream, the accessors, the setters and the
//! projections read and write those five entries only; a matrix given to `from_matrix_unchecked`
//! is trusted to have that structure.
//!
//! API split (the pattern of `UnitQuaternionTrait` / `UnitQuaternionAngleTrait`):
//! - `Perspective3Trait` (`Real` scalar): `from_matrix_unchecked`, the conversions, `inverse`, the
//!   accessors, the projections and the setters that need no trigonometry;
//! - `Perspective3AngleTrait` (`Real` + `Transcendental`): `new`, `fovy`, `set_fovy`;
//! - `Matrix4PerspectiveTrait`: upstream's `Matrix4::new_perspective` (`base/cg.rs`), which is
//!   `Perspective3::new(..).into_inner()`;
//! - `Into<Perspective3, Matrix4>`: upstream's `From<Perspective3> for Matrix4`.
//!
//! Numeric contract (AGENTS.md): every division is ONE correctly rounded `Real::div` (to nearest,
//! ties to even, like `f64 /`), every product one floored fixed-point product, every sum of
//! products one fused kernel; overflow and division by zero panic (upstream's `inf` / `NaN`).
//! Upstream's `relative_eq!(a, b)` assertions use the default tolerances of DESIGN D3
//! (`default_epsilon` = 1 ulp, absolute and relative). Where a formula is reassociated to round
//! fewer times (`znear`, `zfar`, `project_*`), the doc comment says so: the result is closer to
//! upstream's (near-exact `f64`) value than the literal fixed-point transcription, whose variants
//! are kept and measured in the benchmarks of `nalgebra_tests_geometry_projections`.

pub use nalgebra_geometry4::geometry::perspective3::*;
