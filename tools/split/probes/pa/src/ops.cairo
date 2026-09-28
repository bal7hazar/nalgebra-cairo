pub trait Tr<T> { fn tr(self: T) -> u32; }
pub trait MulM<L, R> { type Output; fn mul_mat(self: L, rhs: R) -> Self::Output; }
