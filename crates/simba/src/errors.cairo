//! Panic messages. They are part of the stable API: tests and callers may match on them.

/// The result of an operation does not fit the scalar (Q32.32: `[-2^31, 2^31)`).
pub const OVERFLOW: felt252 = 'simba: overflow';
/// Division, remainder, reciprocal or inverse square root of zero.
pub const DIVISION_BY_ZERO: felt252 = 'simba: division by zero';
/// Square root (or inverse square root, or norm of a negative accumulator) of a negative number.
pub const SQRT_OF_NEGATIVE: felt252 = 'simba: sqrt of negative';
