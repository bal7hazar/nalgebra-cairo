#[derive(Copy, Drop, PartialEq, Debug)]
pub struct A { pub x: u32 }
#[generate_trait]
pub impl AImpl of ATrait { fn new(x: u32) -> A { A { x } } fn get(self: A) -> u32 { self.x } }
pub mod inner {
    // core-trait impl in the type's CRATE but not the type's module
    pub impl MulAInner of Mul<super::A> { fn mul(lhs: super::A, rhs: super::A) -> super::A { super::A { x: lhs.x * rhs.x } } }
}
