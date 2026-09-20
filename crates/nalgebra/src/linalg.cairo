//! Matrix decompositions (upstream `nalgebra::linalg`).
//!
//! Every decomposition here is **closed form or fixed cost**: no convergence loop, no iteration
//! count, no tolerance parameter (DESIGN D6). Gas is therefore a constant of the type, which is
//! what a proof system needs; accuracy is a measured property, reported in the doc comment of
//! each decomposition and checked against `tools/oracle` in the tests.

#[cfg(test)]
mod eigen_test_utils;
#[cfg(test)]
mod oracle_symmetric_eigen;
pub mod symmetric_eigen2;
pub mod symmetric_eigen3;

pub use symmetric_eigen2::{SymmetricEigen2, SymmetricEigen2Trait};
pub use symmetric_eigen3::{SymmetricEigen3, SymmetricEigen3Trait};
