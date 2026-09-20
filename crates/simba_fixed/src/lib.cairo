//! `simba::Real` and `simba::Transcendental` for glam.cairo's shared `fixed::Fixed` scalar.
//!
//! nalgebra.cairo is generic over `Real<T>`; this package lets the physics stack (glam.cairo,
//! rapier.cairo) use nalgebra types with its own scalar, bit-identically (both are Q32.32 on `i64`
//! with floor rounding on every rescale).
