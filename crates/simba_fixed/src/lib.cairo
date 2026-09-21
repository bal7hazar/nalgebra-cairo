//! `simba::Real` and `simba::Transcendental` for glam.cairo's shared `fixed::Fixed` scalar.
//!
//! nalgebra.cairo is generic over `Real<T>`; this package lets the physics stack (glam.cairo,
//! rapier.cairo) use nalgebra types with its own scalar, bit-identically (both are Q32.32 on `i64`
//! with floor rounding on every rescale).
//!
//! ```cairo
//! use fixed::Fixed;
//! use nalgebra::Vector3;
//! use nalgebra::base::vector3::Vector3Trait;
//! use simba_fixed::prelude::*;
//!
//! let v = Vector3 { x: Real::from_int(3), y: Real::from_int(-4), z: Real::<Fixed>::ZERO };
//! assert!(v.norm() == Real::from_int(5));
//! ```
//!
//! # What "bit-identical" covers
//!
//! `FixedReal` relabels its arguments (`raw` to `raw`, free) and calls simba's own kernels, so
//! every nalgebra operation that goes through `Real` — and the library goes through `Real` for
//! every sum of products, DESIGN D3 — returns the same raw values as with `simba::fixed::Fixed`,
//! and panics with the same `'simba: ...'` messages.
//!
//! Three documented differences remain, each asserted case by case in `crate::conformance` and
//! listed here because they are observable:
//!
//! 1. **The foreign operators `/` and `%`.** glam.cairo's own `Div` / `Rem` impls on
//!    `fixed::Fixed` divide toward ZERO and take the truncated remainder (sign of the dividend),
//!    where `simba::fixed::Fixed` floors both; the two agree on exact quotients and on
//!    non-negative operands and differ by one ulp otherwise. This is a property of the SCALAR's
//!    operators only: nalgebra's generic code never reaches them, it divides through `Real::div` /
//!    `Real::rem`, which this impl routes to simba's floor kernels. So `normalize`, `unscale`,
//!    `new_normalize`, the matrix inverses and the decompositions are bit-identical with both
//!    scalars, and a division by zero inside nalgebra panics with `'simba: division by zero'`.
//!    Likewise `Real::recip` is simba's (floor), not `fixed::FixedTrait::recip` (truncating).
//! 2. **Constants.** `Real::<fixed::Fixed>::PI` and friends are simba's floored constants, so that
//!    a formula built from `Real::PI` gives the same result with both scalars; `fixed::PI` is
//!    rounded to nearest and is one ulp higher. Seven of the twenty constants differ by one ulp.
//! 3. **Trigonometry.** `FixedTranscendental` deliberately forwards to glam.cairo's `TrigTrait`
//!    rather than to simba — see its doc comment for why — and the two generated polynomials
//!    differ by a few ulp.
//!
//! `fixed::FixedTrait`'s own helpers (`signum`, `fract`, `recip`, …) are NOT used by this impl;
//! where their semantics differ from simba's, `crate::conformance` records it.

#[cfg(test)]
mod benches;
#[cfg(test)]
mod conformance;
pub mod convert;
#[cfg(test)]
mod oracle_scalar;
pub mod real;
pub mod transcendental;
#[cfg(test)]
mod vectors;

pub use convert::{GlamIntoSimba, SimbaIntoGlam, from_simba, to_simba};
pub use real::FixedReal;
pub use transcendental::FixedTranscendental;

/// `use simba_fixed::prelude::*;`: the two impls, the conversions and the scalar traits they
/// implement (`Real` / `Transcendental` give the method syntax). `fixed::Fixed` itself and its
/// operators come from the `fixed` package.
pub mod prelude {
    pub use simba::fixed::{Wide, WideTrait};
    pub use simba::scalar::{Real, Transcendental};
    pub use crate::convert::{from_simba, to_simba};
    pub use crate::real::FixedReal;
    pub use crate::transcendental::FixedTranscendental;
}
