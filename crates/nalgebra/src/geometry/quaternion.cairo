//! `Quaternion`: a general quaternion `w + x·i + y·j + z·k` (upstream `nalgebra::Quaternion`).
//!
//! - `QuaternionTrait` / `QuaternionImpl`: constructors, parts, algebra, norms, inverse and
//!   interpolation, generic over a `simba::scalar::Real` scalar;
//! - operators `+`, `-`, unary `-` and `*` (the Hamilton product) and the conversions from / to
//!   `Vector4<T>`: their impls live in this module, where the compiler finds them without any
//!   import.
//!
//! Conventions (DESIGN D8, pinned by `test_new_is_w_first` and `test_serde_is_imag_first`):
//!
//! | | order |
//! |---|---|
//! | `new` arguments | `(w, i, j, k)`, like upstream `Quaternion::new` |
//! | fields / `Serde` / `as_vector` | `(i, j, k, w)`, like upstream's `coords: Vector4` |
//! | glam, for glam-cairo's conversions | `Quat::from_xyzw(x, y, z, w)` = `(i, j, k, w)` |
//!
//! The unit quaternion of a 3D rotation is `UnitQuaternion` (`geometry::unit_quaternion`); this
//! type is the general algebra it is built on.
//!
//! - `QuaternionTranscendentalTrait` / `QuaternionTranscendentalImpl`: the transcendental
//!   functions of the general algebra (`exp`, `ln`, `powf`, the trigonometric and hyperbolic
//!   families, the polar decomposition), which additionally need `simba::scalar::Transcendental`
//!   (WP 8.4-P08). They cost one to several transcendental calls each (30 000 to 300 000 gas):
//!   nothing in the physics stack calls them — `UnitQuaternion::from_scaled_axis` IS `exp` of a
//!   pure quaternion and `scaled_axis` its `ln`, both cheaper.
//!
//! The approximate comparisons (`abs_diff_eq`, `relative_eq`, `ulps_eq`) count their tolerances
//! in ulp (DESIGN D3) and, like upstream's `approx` impls, accept `other` OR `-other`
//! component-wise (the double cover of the rotations: `q` and `-q` compare equal).
//!
//! Where upstream's formula yields `NaN` (the logarithm, square root and inverse trigonometric
//! functions of a REAL quaternion, whose imaginary part upstream normalises), these functions
//! panic with `errors::REAL_QUATERNION` instead (PLAN M8 fidelity rules).
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod oracle_ext;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_ext;
pub use nalgebra_static3::geometry::quaternion::*;
// the crate-private helpers the in-crate tests use (`nalgebra_static3::internal`)
#[cfg(test)]
use nalgebra_static3::internal::geometry::quaternion::{QuaternionInternalTrait};
