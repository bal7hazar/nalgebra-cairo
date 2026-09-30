//! In-place inversion of the static square matrices (upstream `SquareMatrix::try_inverse_mut`,
//! `src/linalg/inverse.rs`), on the sizes that have an inverse: the closed forms of
//! `Matrix2/3/4::try_inverse` and the LU inverse of `Matrix6` (`Matrix6LuTrait::try_inverse`).

pub use nalgebra_linalg2::linalg::inverse::*;
pub use nalgebra_linalg3::linalg::inverse::*;
pub use nalgebra_linalg4::linalg::inverse::*;
pub use nalgebra_linalg6::linalg::inverse::*;
