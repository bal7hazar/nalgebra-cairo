//! Panic messages of the `io` module (stable API: changing one is a breaking change).

/// A Matrix Market text that upstream's grammar rejects (no `%%` header line, a missing or
/// malformed shape line): upstream `unwrap`s the `pest` parse result, which panics.
pub const MATRIX_MARKET_SYNTAX: felt252 = 'nalgebra: matrix market syntax';
/// A Matrix Market value whose magnitude does not fit the scalar (upstream parses an `f64`,
/// infinite beyond its range; a fixed-point scalar has no infinity).
pub const VALUE_OUT_OF_RANGE: felt252 = 'nalgebra: value out of range';
