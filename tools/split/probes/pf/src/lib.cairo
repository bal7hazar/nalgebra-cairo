//! Facade probe: glob re-exports of the sub-crates, one module per upstream module.
pub mod base {
    pub use pa::*;
    pub use pb::*;
    pub use pa::errors;
}
pub use pa::*;
pub use pb::*;
pub use pc::*;
pub use pc::mk;
pub mod macros {
    pub macro mk2 {
        ($x:expr) => { $defsite::super::make($x) };
    }
}
pub use macros::mk2;
