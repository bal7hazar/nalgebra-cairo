//! `cumsum` (upstream `sparse::cs_utils::cumsum`) and the crate-internal index kernels of the
//! sparse module.
//!
//! Cairo memory is write-once: upstream's scatters (`workspace[row] += 1`, `res.data.i[shift] =
//! j`) have no direct counterpart. The kernels here reorder `(major, minor, position)` entries by
//! a stable bottom-up MERGE of the runs that are already sorted (the columns of a compressed
//! matrix, the ascending stretches of a triplet list), then GATHER the values through the
//! positions (span reads are random access). Every loop walks spans front to back.

pub use nalgebra_sparse::sparse::cs_utils::*;
