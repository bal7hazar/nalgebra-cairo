use pa::{A, MulM};
#[derive(Copy, Drop, PartialEq, Debug)]
pub struct B { pub x: u32 }
#[generate_trait]
pub impl BImpl of BTrait { fn new(x: u32) -> B { B { x } } }
pub impl MulAB of MulM<A, B> { type Output = B; fn mul_mat(self: A, rhs: B) -> B { B { x: self.x * rhs.x } } }
pub impl SubA of Sub<A> { fn sub(lhs: A, rhs: A) -> A { A { x: lhs.x - rhs.x } } }
// core trait with two nalgebra-like types, placed in the module of the SECOND type (B)
pub impl IntoAB2 of Into<A, B> { fn into(self: A) -> B { B { x: self.x } } }
pub impl AEqB of PartialEq<B> { fn eq(lhs: @B, rhs: @B) -> bool { *lhs.x == *rhs.x } }
