pub mod shapes;
pub mod ops;
pub use shapes::{A, ATrait};
pub use ops::{MulM, Tr};
pub mod errors { pub const E: felt252 = 'a'; }
#[cfg(feature: 'extra')]
pub fn extra() -> u32 { 7 }
