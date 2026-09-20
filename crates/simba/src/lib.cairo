//! Scalar abstraction and Q32.32 fixed-point numbers.
//!
//! Counterpart of the Rust `simba` crate for nalgebra.cairo. See `docs/DESIGN.md` (D2, D3).
//!
//! - `fixed`: the Q32.32 `Fixed` type, its operators, helpers, fused kernels and `Wide`
//!   accumulator;
//! - `scalar`: the generic `Real` trait (implemented by `Fixed`) and `Transcendental`;
//! - `errors`: stable panic messages;
//! - `prelude`: everything needed to use `Fixed` with method syntax.

pub mod errors;
pub mod fixed;
pub mod scalar;

/// `use simba::prelude::*;`: the `Fixed` type, its `Wide` accumulator and the scalar traits (method
/// syntax). Operator and conversion impls need no import: they are re-exported by the module of
/// `Fixed`, where the compiler looks them up.
pub mod prelude {
    pub use crate::fixed::types::Fixed;
    pub use crate::fixed::wide::{Wide, WideTrait};
    pub use crate::scalar::{FixedReal, Real, Transcendental};
}
