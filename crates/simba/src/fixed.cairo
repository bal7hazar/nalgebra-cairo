//! Q32.32 fixed-point numbers on a native `i64` (DESIGN D2).
//!
//! - `types`: the `Fixed` struct and its constants;
//! - `ops`: operators (`+ - * / %`, unary `-`, comparisons, compound assignments);
//! - `convert`: integer conversions;
//! - `math`: abs, signum, min, max, clamp, rounding to integers, recip, sqrt, inv_sqrt;
//! - `fused`: single-rounding sums of products (`sum_prod*`, `diff_prod`, `mul_add`, `norm*`);
//! - `transcendental`: sin, cos, tan and their inverses;
//! - `wide`: explicit unscaled accumulator for longer sums;
//! - `kernels`: raw `i64` kernels, the only user of `core::internal::bounded_int`.
//!
//! Method syntax and generic code go through `simba::scalar::Real`.

pub mod convert;
pub mod fused;
pub mod kernels;
pub mod math;
pub mod ops;
#[cfg(test)]
mod oracle_scalar;
#[cfg(test)]
mod tests_generated;
pub mod transcendental;
pub mod types;
pub mod wide;
pub use types::Fixed;
pub use wide::{Wide, WideTrait};
