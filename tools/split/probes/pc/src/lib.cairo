pub mod ext;
pub use ext::*;
pub fn make(x: u32) -> pa::A { pa::A { x } }
pub macro mk {
    ($x:expr) => { $defsite::make($x) };
}
