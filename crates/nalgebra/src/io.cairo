//! Parsers of matrix formats (upstream `nalgebra::io`): the Matrix Market coordinate format into
//! a `CsMatrix`. Behind the Scarb feature `io` (in `default`, DESIGN D9; it enables `sparse`).
//!
//! Upstream's `cs_matrix_from_matrix_market(path)` reads a file: a Cairo program has no file
//! system, so only the `_str` form exists here (the content is passed as a `ByteArray`).

pub mod errors;
pub mod matrix_market;

pub use matrix_market::cs_matrix_from_matrix_market_str;
