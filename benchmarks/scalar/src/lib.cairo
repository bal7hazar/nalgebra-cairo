//! Gas micro-benchmarks: fixed-point scalar representations for nalgebra.cairo.
//! See README.md for the methodology and the results.
pub mod algo;
pub mod bi64;
pub mod felt_fixed;
pub mod generic;
pub mod i128q64;
pub mod i32q16;
pub mod i64q32;
pub mod signmag32;
pub mod signmag64;

#[cfg(test)]
mod tests;
