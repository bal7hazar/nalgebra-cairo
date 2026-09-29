//! Internal, no stability promise: the crate-private items of `linalg::exp` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// Panic message of the LU solve of a Padé denominator with an exactly zero pivot (upstream
/// unwraps `LU::solve`).
pub const SINGULAR_PADE: felt252 = 'nalgebra: singular Pade denom';
