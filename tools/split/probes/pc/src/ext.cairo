use pa::{A, Tr, MulM};
use pb::B;
pub impl TrB of Tr<B> { fn tr(self: B) -> u32 { self.x + 100 } }
pub impl AddA of Add<A> { fn add(lhs: A, rhs: A) -> A { A { x: lhs.x + rhs.x } } }
#[generate_trait]
pub impl BExtImpl of BExtTrait { fn twice(self: B) -> u32 { self.x * 2 } }
pub impl MulBA of MulM<B, A> { type Output = A; fn mul_mat(self: B, rhs: A) -> A { A { x: self.x * rhs.x + 1 } } }
