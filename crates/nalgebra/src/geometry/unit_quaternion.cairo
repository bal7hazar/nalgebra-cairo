//! `UnitQuaternion`: a 3D rotation as a quaternion of unit norm (upstream
//! `nalgebra::UnitQuaternion`).
//!
//! - `UnitQuaternionTrait` / `UnitQuaternionImpl`: construction, products, transforms, conversions
//!   to and from a rotation matrix, renormalisation and the integration step — everything that
//!   needs no trigonometry;
//! - `UnitQuaternionAngleTrait` / `UnitQuaternionAngleImpl`: axis-angle, Euler angles, `slerp` and
//!   `powf`, which additionally need `simba::scalar::Transcendental`;
//! - the operators `*` (composition) and unary `-` live in this module, where the compiler finds
//!   them without any import.
//!
//! The split in two traits mirrors `Vector3Trait` / `Vector3AngleTrait`: a scalar may implement
//! `Real` without `Transcendental`, and then the whole rapier hot path is still available
//! (`q * q`, `transform_vector`, `inverse_transform_vector`, `append_axisangle_linearized`,
//! `renormalize_fast`) — it needs no trigonometry at all.
//!
//! Upstream this type is `Unit<Quaternion<T>>`; here it is a dedicated wrapper struct with the same
//! invariant, so that the operators and the rotation methods live in this module (a Cairo operator
//! impl is only found in the module of its type, and `Unit` lives in `base::unit`). The invariant
//! `|q| = 1` is a CONTRACT: `new_unchecked` does not check it, and the quaternions produced here
//! have a norm of `1` within about 2 ulp, not exactly `1` (fixed point cannot do better).
//! `renormalize_fast` (one Newton step, no square root) brings a drifted rotation back.
//!
//! `q` and `-q` are the same rotation. No operation normalises the sign, except `slerp` (shortest
//! arc) and `axis` / `angle` / `scaled_axis` / `axis_angle` (which report the `w >= 0`
//! representative, i.e. an angle in `[0, π]`), exactly like upstream.
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
pub use nalgebra_static3::geometry::unit_quaternion::*;
// the crate-private helpers the in-crate tests use (`nalgebra_static3::internal`)
#[cfg(test)]
use nalgebra_static3::internal::geometry::unit_quaternion::{
    UnitQuaternionAngleInternalTrait, UnitQuaternionInternalTrait,
};
