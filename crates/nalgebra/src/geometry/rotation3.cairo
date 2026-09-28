//! `Rotation3`: a 3D rotation stored as a 3x3 orthonormal matrix (upstream `nalgebra::Rotation3`).
//!
//! - `Rotation3Trait` / `Rotation3Impl`: constructors, transforms, composition, conversions to and
//!   from `UnitQuaternion`, renormalisation — everything that needs no trigonometry;
//! - `Rotation3AngleTrait` / `Rotation3AngleImpl`: axis-angle and Euler-angle constructors and
//!   extractors, which additionally need `simba::scalar::Transcendental`;
//! - the operator `*` (composition) lives in this module, where the compiler finds it without any
//!   import.
//!
//! The matrix form is the right representation when several vectors are transformed with the same
//! rotation: `transform_vector` is one `Matrix3 * Vector3` (9 products, 6 650 gas) against 15
//! products (23 230) for `UnitQuaternion::transform_vector`, and building the matrix from a
//! quaternion costs 23 530 — so the break-even is TWO vectors per step (measured,
//! `bench_unit_quaternion_transform_vector_x2__*`). Composition is the other way round: 23 310 for
//! a matrix product against 11 860 for a Hamilton product, and a matrix composed repeatedly drifts
//! out of orthonormality, which `renormalize` (upstream's closed-form polar factor, about 530 000
//! gas) has to fix.
//!
//! The invariant (orthonormal, determinant +1) is a CONTRACT, like upstream's
//! `from_matrix_unchecked`: nothing checks it, and `inverse` is implemented as the transpose.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

#[cfg(test)]
mod benches;
#[cfg(test)]
mod tests;
pub use nalgebra_static3::geometry::rotation3::*;
// the crate-private helpers the in-crate tests use (`nalgebra_static3::internal`)
#[cfg(test)]
use nalgebra_static3::internal::geometry::rotation3::{Rotation3InternalTrait};
