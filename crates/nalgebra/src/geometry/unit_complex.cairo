//! `UnitComplex`: a 2D rotation stored as a unit complex number (upstream
//! `nalgebra::UnitComplex`, which is `Unit<Complex<T>>`).
//!
//! - `UnitComplexTrait` / `UnitComplexImpl`: construction, accessors, composition, transforms,
//!   conversions and renormalization — everything that is algebraic, hence available for any
//!   `simba::scalar::Real` scalar;
//! - `UnitComplexAngleTrait` / `UnitComplexAngleImpl`: the operations that go through an angle
//!   (`new`, `angle`, `powf`, `slerp`, ...), which additionally need
//!   `simba::scalar::Transcendental`;
//! - `a * b` (composition of two rotations): its impl lives in this module, where the compiler
//!   finds it without any import.
//!
//! The raw pair `(re, im) = (cos θ, sin θ)` is stored directly instead of wrapping a
//! `Unit<Vector2<T>>`: a rotation is not a vector (it composes with `*`, not with `+`), and the
//! extra layer costs an indirection in every formula for nothing. Nothing enforces the invariant
//! `re² + im² = 1`: build with `new` / `rotation_between` / `from_rotation_matrix`, or with
//! `from_cos_sin_unchecked` when the pair is known to be normalized. Rotations built from an angle
//! have a norm of `1` within a few ulp, not exactly `1` (fixed point cannot do better);
//! `renormalize` / `renormalize_fast` bring a drifted pair back.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle_ext;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_ext;
pub use nalgebra_static3::geometry::unit_complex::*;
// the crate-private helpers the in-crate tests use (`nalgebra_static3::internal`)
#[cfg(test)]
use nalgebra_static3::internal::geometry::unit_complex::{
    UnitComplexAngleInternalTrait, UnitComplexInternalTrait,
};
