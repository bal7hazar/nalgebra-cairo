//! The linalg of `nalgebra_static6_wide` (upstream `nalgebra::linalg`): `lu::lu6` (`Lu6`, the LU
//! factorisation of `Matrix6`, which `Matrix6::is_invertible` runs) and `Perm6` (`lu`); the rest of
//! the module is above (the facade `nalgebra` re-exports every item at its 0.1.0 path).

pub mod lu;

// the generated `Matrix6` names `crate::linalg::Matrix6LuTrait` (0.1.0's re-export)
pub use lu::lu6::Matrix6LuTrait;
