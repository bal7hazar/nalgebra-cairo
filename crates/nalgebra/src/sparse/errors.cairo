//! Panic messages of the sparse module (stable API: changing one is a breaking change). The shape
//! and index errors reuse `base::errors` (`nalgebra: dimension mismatch`, `nalgebra: index out
//! of bounds`).

pub use nalgebra_sparse::sparse::errors::*;
