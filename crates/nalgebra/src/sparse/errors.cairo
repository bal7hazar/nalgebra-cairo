//! Panic messages of the sparse module (stable API: changing one is a breaking change). The shape
//! and index errors reuse `base::errors` (`nalgebra: dimension mismatch`, `nalgebra: index out
//! of bounds`).

/// A square matrix was expected: the triangular solves (upstream: "The matrix must be square.")
/// and `CsCholesky::new_symbolic` (upstream: "The matrix `m` must be square to compute its
/// elimination tree.").
pub const NOT_SQUARE: felt252 = 'nalgebra: matrix not square';
/// `from_triplet` with index / value slices of different lengths (upstream: a bare `assert!`).
pub const TRIPLET_LENGTHS: felt252 = 'nalgebra: triplet lengths';
/// `CsCholesky::decompose_*` with fewer values than the analysed pattern has entries (upstream:
/// "The set of values is too small.").
pub const VALUES_TOO_SHORT: felt252 = 'nalgebra: values too short';
